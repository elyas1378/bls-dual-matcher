#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional RapidFuzz
try:
    from rapidfuzz import fuzz, process
    _HAVE_RAPIDFUZZ = True
except Exception:
    fuzz = None
    process = None
    _HAVE_RAPIDFUZZ = False

# Optional LLM clients
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None


def normalize_text(text: str, fold_umlauts: bool = True) -> str:
    s = str(text).strip().lower()
    s = unicodedata.normalize("NFKC", s)

    s = s.replace("ß", "ss")
    if fold_umlauts:
        s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")

    # light canonicalization
    s = s.replace("yoghurt", "joghurt")

    # remove quantities/units
    s = re.sub(
        r"\b\d+([.,]\d+)?\s*(g|kg|mg|ml|l|stk|stueck|stück|scheibe(n)?|pack(ung)?|becher|dose(n)?)\b",
        " ",
        s,
    )

    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_json_dict(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    csv_name: str
    code_col: str
    name_col: str
    lookup_user: str
    lookup_unique: str
    lookup_ambiguous: str


class DualFoodMatcher:
    # Retrieval params
    FOLD_UMLAUTS = True
    CHAR_MIN_DF = 1
    CHAR_NGRAM_RANGE = (3, 5)
    WORD_NGRAM_RANGE = (1, 2)
    WORD_MAX_FEATURES = 80000

    # Candidate sizes
    TOP_K_RETURN = 200
    RAPIDFUZZ_POOL = 1000
    LLM_TOP_N = 10  # you wanted ~10

    # Fusion weights
    W_CHAR = 0.55
    W_RF = 0.30
    W_WORD = 0.15

    # --- Behavior knobs (important) ---
    ENABLE_LLM_REWRITE = True

    # Lookup fuzzy match threshold (0-100 RapidFuzz score)
    LOOKUP_FUZZY_MIN = 92

    # Decide if retrieval is "confident enough" to skip LLM
    # Score is 0-1 (your fused score). Tune if needed.
    RETRIEVAL_SKIP_LLM_MIN = 0.78
    RETRIEVAL_MARGIN_MIN = 0.06

    # Rewrite only when retrieval looks weak
    RETRIEVAL_REWRITE_IF_TOP_BELOW = 0.55

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).expanduser().resolve()

        # Streamlit Cloud runs from repo root. We keep repo-root aware paths.
        webapp_dir = self.project_root / "webapp"

        # outputs always in webapp/outputs (that’s where your JSONs live)
        self.outputs = webapp_dir / "outputs"
        self.outputs.mkdir(parents=True, exist_ok=True)

        # data_processed sits at repo root
        self.data_processed = self.project_root / "data_processed"

        # LLM init
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self.llm = None
        self._init_llm()

        self.cfg_302 = DatasetConfig(
            key="bls302",
            csv_name="BLS_302_slim.csv",
            code_col="bls302_code",
            name_col="name_de",
            lookup_user="lookup_user.json",
            lookup_unique="lookup_unique.json",
            lookup_ambiguous="lookup_ambiguous.json",
        )

        self.cfg_40 = DatasetConfig(
            key="bls40",
            csv_name="BLS_40_slim.csv",
            code_col="bls4_code",
            name_col="name_de",
            lookup_user="lookup_user_bls40.json",
            lookup_unique="lookup_unique_bls40.json",
            lookup_ambiguous="lookup_ambiguous_bls40.json",
        )

        self.ds: Dict[str, dict] = {}
        for cfg in [self.cfg_302, self.cfg_40]:
            self.ds[cfg.key] = self._load_dataset(cfg)

        print(f"✓ DualFoodMatcher ready | outputs={self.outputs} | data_processed={self.data_processed}")
        print(f"✓ LLM provider={self.llm_provider} | LLM available={self.llm is not None} | RapidFuzz={_HAVE_RAPIDFUZZ}")

    # ----------------------------
    # LLM init
    # ----------------------------
    def _init_llm(self):
        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and OpenAI is not None:
                self.llm = OpenAI(api_key=api_key)
                print("✓ OpenAI API initialized")
            else:
                print("⚠️  OpenAI unavailable (missing key or package)")
        elif self.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key and Anthropic is not None:
                self.llm = Anthropic(api_key=api_key)
                print("✓ Anthropic API initialized")
            else:
                print("⚠️  Anthropic unavailable (missing key or package)")
        else:
            print(f"⚠️  Unknown LLM_PROVIDER='{self.llm_provider}', disabling LLM")
            self.llm = None

    # ----------------------------
    # Dataset loader
    # ----------------------------
    def _load_dataset(self, cfg: DatasetConfig) -> dict:
        csv_path = (self.data_processed / cfg.csv_name).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"[{cfg.key}] Missing CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        for col in [cfg.code_col, cfg.name_col]:
            if col not in df.columns:
                raise ValueError(f"[{cfg.key}] CSV missing column '{col}'. Found: {list(df.columns)}")

        df[cfg.code_col] = df[cfg.code_col].astype(str)
        df[cfg.name_col] = df[cfg.name_col].fillna("").astype(str)

        df["name_norm"] = df[cfg.name_col].map(lambda x: normalize_text(x, fold_umlauts=self.FOLD_UMLAUTS))
        names_norm = df["name_norm"].tolist()

        # lookups (from webapp/outputs)
        lu_user = _load_json_dict(self.outputs / cfg.lookup_user)
        lu_unique = _load_json_dict(self.outputs / cfg.lookup_unique)
        lu_amb = _load_json_dict(self.outputs / cfg.lookup_ambiguous)

        # precompute lookup key list for fuzzy matching
        lookup_keys = sorted(set(list(lu_user.keys()) + list(lu_unique.keys())))
        lookup_key_to_code = {**lu_unique, **lu_user}  # user overrides unique if both exist

        # retrieval indices
        vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=self.CHAR_NGRAM_RANGE, min_df=self.CHAR_MIN_DF)
        mat_char = vec_char.fit_transform(names_norm)

        vec_word = TfidfVectorizer(ngram_range=self.WORD_NGRAM_RANGE, min_df=2, max_features=self.WORD_MAX_FEATURES)
        mat_word = vec_word.fit_transform(names_norm)

        return {
            "cfg": cfg,
            "df": df,
            "names_norm": names_norm,
            "lookup_user": lu_user,
            "lookup_unique": lu_unique,
            "lookup_ambiguous": lu_amb,
            "lookup_keys": lookup_keys,
            "lookup_key_to_code": lookup_key_to_code,
            "vec_char": vec_char,
            "mat_char": mat_char,
            "vec_word": vec_word,
            "mat_word": mat_word,
        }

    def _code_to_name(self, dataset_key: str, code: str) -> str:
        d = self.ds[dataset_key]
        cfg = d["cfg"]
        df = d["df"]
        row = df[df[cfg.code_col] == str(code)]
        return row[cfg.name_col].values[0] if len(row) else ""

    # ----------------------------
    # Lookup (exact + fuzzy)
    # ----------------------------
    def _lookup_hit(self, dataset_key: str, raw_text: str) -> Optional[Tuple[str, str, str]]:
        d = self.ds[dataset_key]
        key = normalize_text(raw_text, fold_umlauts=self.FOLD_UMLAUTS)
        if not key:
            return None

        # exact user lookup first
        if key in d["lookup_user"]:
            code = str(d["lookup_user"][key]).strip()
            return ("user_lookup_exact", code, self._code_to_name(dataset_key, code))

        # exact unique lookup
        if key in d["lookup_unique"]:
            code = str(d["lookup_unique"][key]).strip()
            return ("unique_lookup_exact", code, self._code_to_name(dataset_key, code))

        # fuzzy lookup: match query against lookup keys (human-labeled universe)
        keys = d.get("lookup_keys", [])
        if not keys:
            return None

        best_key = None
        best_score = 0

        if _HAVE_RAPIDFUZZ and process is not None:
            # token-based fuzzing tends to behave better on short foods
            match = process.extractOne(
                key,
                keys,
                scorer=fuzz.WRatio,
            )
            if match:
                best_key, best_score, _ = match
        else:
            # fallback: no rapidfuzz => no fuzzy lookup
            return None

        if best_key and best_score >= self.LOOKUP_FUZZY_MIN:
            code = str(d["lookup_key_to_code"].get(best_key, "")).strip()
            if code:
                src = "user_lookup_fuzzy" if best_key in d["lookup_user"] else "unique_lookup_fuzzy"
                return (src, code, self._code_to_name(dataset_key, code))

        return None

    # ----------------------------
    # Retrieval candidates
    # ----------------------------
    def generate_candidates(self, dataset_key: str, query: str, top_k: int = None) -> List[dict]:
        if top_k is None:
            top_k = self.TOP_K_RETURN

        d = self.ds[dataset_key]
        cfg = d["cfg"]

        q = normalize_text(query, fold_umlauts=self.FOLD_UMLAUTS)
        if not q:
            return []

        q_char = d["vec_char"].transform([q])
        sim_char = cosine_similarity(q_char, d["mat_char"])[0]

        q_word = d["vec_word"].transform([q])
        sim_word = cosine_similarity(q_word, d["mat_word"])[0]

        rf = np.zeros_like(sim_char)
        if _HAVE_RAPIDFUZZ:
            pool = min(self.RAPIDFUZZ_POOL, len(sim_char))
            pool_idx = np.argsort(sim_char)[::-1][:pool]
            names_norm = d["names_norm"]
            for idx in pool_idx:
                rf[idx] = fuzz.WRatio(q, names_norm[idx]) / 100.0

        score = self.W_CHAR * sim_char + self.W_WORD * sim_word + self.W_RF * rf
        top_idx = np.argsort(score)[::-1][:top_k]

        df = d["df"]
        candidates = []
        for rank, idx in enumerate(top_idx, 1):
            candidates.append({
                "rank": rank,
                "code": df.iloc[idx][cfg.code_col],
                "name": df.iloc[idx][cfg.name_col],
                "score": float(score[idx]),
            })
        return candidates

    # ----------------------------
    # LLM helpers
    # ----------------------------
    def llm_rewrite_query(self, dataset_key: str, food_text: str) -> str:
        if not (self.llm and self.ENABLE_LLM_REWRITE):
            return food_text

        cfg = self.ds[dataset_key]["cfg"]
        label = "BLS 3.02" if cfg.key == "bls302" else "BLS 4.0"

        prompt = f"""You help a retrieval system search a German food database ({label}).
Given a user food text, output a SHORT rewritten German query using common food words likely in the database.
Rules:
- Keep it under 8 words.
- Prefer generic food nouns over brands.
- If term is "skyr", rewrite to "joghurt eiweissreich" or "quark eiweissreich".
- Return ONLY the rewritten query.

User text: "{food_text}"
"""

        try:
            if self.llm_provider == "openai":
                resp = self.llm.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=40,
                )
                out = resp.choices[0].message.content.strip()
                return out if out else food_text

            if self.llm_provider == "anthropic":
                resp = self.llm.messages.create(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                    max_tokens=40,
                    messages=[{"role": "user", "content": prompt}],
                )
                out = resp.content[0].text.strip()
                return out if out else food_text

        except Exception:
            return food_text

        return food_text

    def ask_llm_choose(self, dataset_key: str, food_text: str, candidates: List[dict]) -> Tuple[str, str, int]:
        if not candidates:
            return "", "", 0

        # if no llm, fallback to best candidate
        if not self.llm:
            best = candidates[0]
            return best["code"], best["name"], 65

        cfg = self.ds[dataset_key]["cfg"]
        label = "BLS 3.02" if cfg.key == "bls302" else "BLS 4.0"

        cand_text = "\n".join(
            [f"{i}. {c['code']} - {c['name']}" for i, c in enumerate(candidates[:self.LLM_TOP_N], 1)]
        )

        prompt = f"""You are matching a user food to the closest {label} item.
Pick the single BEST code from the candidates.

User food: "{food_text}"

Candidates:
{cand_text}

Return ONLY the code. No explanation.
"""

        try:
            if self.llm_provider == "openai":
                response = self.llm.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0,
                )
                chosen = response.choices[0].message.content.strip()
            elif self.llm_provider == "anthropic":
                response = self.llm.messages.create(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                    max_tokens=20,
                    messages=[{"role": "user", "content": prompt}],
                )
                chosen = response.content[0].text.strip()
            else:
                chosen = ""

            for c in candidates[: self.LLM_TOP_N]:
                if c["code"] == chosen:
                    return c["code"], c["name"], 85

            best = candidates[0]
            return best["code"], best["name"], 70

        except Exception:
            best = candidates[0]
            return best["code"], best["name"], 60

    # ----------------------------
    # Core matching
    # ----------------------------
    def match_one(self, dataset_key: str, food_text: str) -> dict:
        raw = str(food_text).strip()

        # 1) Lookup (human truth)
        lh = self._lookup_hit(dataset_key, raw)
        if lh is not None:
            src, code, name = lh
            # show a few retrieval candidates too (for transparency)
            cands = self.generate_candidates(dataset_key, raw, top_k=50)
            return {
                "food": raw,
                "code": code,
                "name": name,
                "confidence": 100,
                "source": "lookup",
                "match_source": src,
                "rewritten_query": None,
                "candidates": cands[:5],
            }

        # 2) Retrieval (no rewrite first)
        candidates = self.generate_candidates(dataset_key, raw, top_k=50)
        if not candidates:
            return {
                "food": raw,
                "code": "",
                "name": "",
                "confidence": 0,
                "source": "none",
                "match_source": "none",
                "rewritten_query": None,
                "candidates": [],
            }

        top1 = candidates[0]["score"]
        top2 = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = top1 - top2

        # If retrieval is clearly confident, do NOT waste LLM calls
        if top1 >= self.RETRIEVAL_SKIP_LLM_MIN and margin >= self.RETRIEVAL_MARGIN_MIN:
            best = candidates[0]
            conf = int(min(95, max(70, round(70 + 25 * best["score"]))))
            return {
                "food": raw,
                "code": best["code"],
                "name": best["name"],
                "confidence": conf,
                "source": "retrieval",
                "match_source": "retrieval_best",
                "rewritten_query": None,
                "candidates": candidates[:5],
            }

        # 3) If LLM not available, return best retrieval anyway
        if not self.llm:
            best = candidates[0]
            return {
                "food": raw,
                "code": best["code"],
                "name": best["name"],
                "confidence": 65,
                "source": "hybrid",
                "match_source": "retrieval_best_no_llm",
                "rewritten_query": None,
                "candidates": candidates[:5],
            }

        # 4) If retrieval looks weak, rewrite then re-retrieve
        rewritten = None
        if top1 < self.RETRIEVAL_REWRITE_IF_TOP_BELOW:
            rewritten_try = self.llm_rewrite_query(dataset_key, raw)
            if rewritten_try and rewritten_try.strip() and rewritten_try.strip().lower() != raw.strip().lower():
                rewritten = rewritten_try
                candidates = self.generate_candidates(dataset_key, rewritten, top_k=50)

        # 5) LLM chooses among top-N candidates
        code, name, conf = self.ask_llm_choose(dataset_key, raw, candidates)
        return {
            "food": raw,
            "code": code,
            "name": name,
            "confidence": conf,
            "source": "llm",
            "match_source": "choose_top_candidates" if rewritten is None else "rewrite_then_choose",
            "rewritten_query": rewritten,
            "candidates": candidates[:5],
        }

    def match_both(self, food_text: str) -> dict:
        return {
            "food": str(food_text).strip(),
            "bls302": self.match_one("bls302", food_text),
            "bls40": self.match_one("bls40", food_text),
        }

    def add_to_lookup(self, dataset_key: str, food_text: str, code: str) -> None:
        d = self.ds[dataset_key]
        cfg = d["cfg"]
        key = normalize_text(food_text, fold_umlauts=self.FOLD_UMLAUTS)
        if not key:
            return

        d["lookup_user"][key] = str(code).strip()
        _save_json_dict(self.outputs / cfg.lookup_user, d["lookup_user"])

        # refresh fuzzy lookup caches
        d["lookup_keys"] = sorted(set(list(d["lookup_user"].keys()) + list(d["lookup_unique"].keys())))
        d["lookup_key_to_code"] = {**d["lookup_unique"], **d["lookup_user"]}

    def get_stats(self) -> dict:
        def one(k: str) -> dict:
            d = self.ds[k]
            cfg = d["cfg"]
            return {
                "csv": str((self.data_processed / cfg.csv_name).resolve()),
                "items": int(len(d["df"])),
                "lookup_user": len(d["lookup_user"]),
                "lookup_unique": len(d["lookup_unique"]),
                "lookup_ambiguous": len(d["lookup_ambiguous"]),
            }

        return {
            "outputs_dir": str(self.outputs),
            "data_processed_dir": str(self.data_processed),
            "llm_provider": self.llm_provider,
            "llm_available": self.llm is not None,
            "rapidfuzz_available": _HAVE_RAPIDFUZZ,
            "bls302": one("bls302"),
            "bls40": one("bls40"),
        }
