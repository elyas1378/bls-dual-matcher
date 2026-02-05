#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional RapidFuzz
try:
    from rapidfuzz import fuzz
    _HAVE_RAPIDFUZZ = True
except Exception:
    fuzz = None
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

    s = s.replace("yoghurt", "joghurt")

    # remove common quantity units
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
    generic_block: set


class DualFoodMatcher:
    # Retrieval params
    FOLD_UMLAUTS = True
    CHAR_MIN_DF = 1
    CHAR_NGRAM_RANGE = (3, 5)
    WORD_NGRAM_RANGE = (1, 2)
    WORD_MAX_FEATURES = 80000

    # How many candidates we compute vs show vs give to LLM
    TOP_K_RETURN = 200
    UI_TOP_K = 10
    LLM_TOP_N = 10
    RAPIDFUZZ_POOL = 1000

    # Score fusion
    W_CHAR = 0.55
    W_RF = 0.30
    W_WORD = 0.15

    # LLM behavior
    ENABLE_LLM_REWRITE = True

    # Strength thresholds (tune later if needed)
    STRONG_SCORE = 0.72   # if top-1 >= this and margin decent, retrieval is strong
    STRONG_MARGIN = 0.05  # top1 - top2
    WEAK_SCORE = 0.45     # if top-1 < this, retrieval is weak/garbage

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).expanduser().resolve()

        # --- webapp-aware pathing ---
        webapp_dir = self.project_root / "webapp"
        self.outputs = (webapp_dir / "outputs") if (webapp_dir / "outputs").exists() else (self.project_root / "outputs")
        self.outputs.mkdir(parents=True, exist_ok=True)

        if (self.project_root / "data_processed").exists():
            self.data_processed = self.project_root / "data_processed"
        elif (webapp_dir / "data_processed").exists():
            self.data_processed = webapp_dir / "data_processed"
        else:
            self.data_processed = self.project_root / "data_processed"

        # LLM init
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self.llm = None
        self._init_llm()

        # Generic words are NOT allowed to block human lookup_user.
        # We only use this to avoid auto-matching from lookup_unique for generic terms.
        generic_block = {
            "skyr", "milch", "kaese", "käse", "salat", "zwiebel", "quark", "joghurt",
            "gemuese", "gemüse", "apfel", "nudeln", "spinat", "haehnchen", "hähnchen",
            "sosse", "soße", "oel", "öl", "wasser"
        }

        self.cfg_302 = DatasetConfig(
            key="bls302",
            csv_name="BLS_302_slim.csv",
            code_col="bls302_code",
            name_col="name_de",
            lookup_user="lookup_user.json",
            lookup_unique="lookup_unique.json",
            lookup_ambiguous="lookup_ambiguous.json",
            generic_block=generic_block,
        )

        self.cfg_40 = DatasetConfig(
            key="bls40",
            csv_name="BLS_40_slim.csv",
            code_col="bls4_code",
            name_col="name_de",
            lookup_user="lookup_user_bls40.json",
            lookup_unique="lookup_unique_bls40.json",
            lookup_ambiguous="lookup_ambiguous_bls40.json",
            generic_block=generic_block,
        )

        self.ds: Dict[str, Dict[str, Any]] = {}
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

        # lookups (from outputs dir)
        lu_user = _load_json_dict(self.outputs / cfg.lookup_user)
        lu_unique = _load_json_dict(self.outputs / cfg.lookup_unique)
        lu_amb = _load_json_dict(self.outputs / cfg.lookup_ambiguous)

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

    def _is_generic(self, dataset_key: str, raw_text: str) -> bool:
        d = self.ds[dataset_key]
        cfg = d["cfg"]
        key = normalize_text(raw_text, fold_umlauts=self.FOLD_UMLAUTS)
        if not key:
            return False
        return key in {normalize_text(x, fold_umlauts=self.FOLD_UMLAUTS) for x in cfg.generic_block}

    # ----------------------------
    # Lookup logic (FIXED to obey humans)
    # ----------------------------
    def _lookup_hit(self, dataset_key: str, raw_text: str) -> Optional[Tuple[str, str, str]]:
        """
        Authoritative behavior:
        - lookup_user ALWAYS wins (even for generic terms)
        - lookup_unique can be skipped for generic terms (optional safety)
        """
        d = self.ds[dataset_key]
        cfg = d["cfg"]
        key = normalize_text(raw_text, fold_umlauts=self.FOLD_UMLAUTS)
        if not key:
            return None

        # 1) HUMAN (authoritative)
        if key in d["lookup_user"]:
            code = str(d["lookup_user"][key]).strip()
            return ("user_lookup", code, self._code_to_name(dataset_key, code))

        # 2) Optional: skip auto-unique for generic terms (prevents overconfident auto-assignments)
        if self._is_generic(dataset_key, raw_text):
            return None

        if key in d["lookup_unique"]:
            code = str(d["lookup_unique"][key]).strip()
            return ("unique_lookup", code, self._code_to_name(dataset_key, code))

        return None

    def _ambiguous_candidates_from_lookup(self, dataset_key: str, raw_text: str) -> List[dict]:
        d = self.ds[dataset_key]
        key = normalize_text(raw_text, fold_umlauts=self.FOLD_UMLAUTS)
        if key in d["lookup_ambiguous"]:
            lst = d["lookup_ambiguous"][key]
            if isinstance(lst, list) and len(lst):
                out = []
                for i, item in enumerate(lst, 1):
                    code = str(item.get("code", "")).strip()
                    if not code:
                        continue
                    out.append({
                        "rank": i,
                        "code": code,
                        "name": self._code_to_name(dataset_key, code),
                        "score": 0.0,
                        "_from": "lookup_ambiguous",
                        "_count": int(item.get("n", 0)) if str(item.get("n", "")).isdigit() else 0,
                    })
                return out
        return []

    # ----------------------------
    # Retrieval
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

    def _candidate_strength(self, candidates: List[dict]) -> Tuple[float, float, bool, bool]:
        """
        Returns:
        - top1 score
        - margin (top1 - top2)
        - is_strong
        - is_weak
        """
        if not candidates:
            return 0.0, 0.0, False, True
        s1 = float(candidates[0].get("score", 0.0) or 0.0)
        s2 = float(candidates[1].get("score", 0.0) or 0.0) if len(candidates) > 1 else 0.0
        margin = s1 - s2
        is_strong = (s1 >= self.STRONG_SCORE) and (margin >= self.STRONG_MARGIN)
        is_weak = (s1 < self.WEAK_SCORE)
        return s1, margin, is_strong, is_weak

    # ----------------------------
    # LLM
    # ----------------------------
    def llm_rewrite_query(self, dataset_key: str, food_text: str, mode: str = "rewrite") -> str:
        """
        mode:
          - "rewrite": normal rewrite
          - "rescue": rewrite more aggressively if retrieval is weak
        """
        if not (self.llm and self.ENABLE_LLM_REWRITE):
            return food_text

        cfg = self.ds[dataset_key]["cfg"]
        label = "BLS 3.02" if cfg.key == "bls302" else "BLS 4.0"

        if mode == "rescue":
            extra = (
                "- Be more general if the input is very specific.\n"
                "- Remove brand names, packaging words, and adjectives.\n"
                "- If unclear, output the most likely base food noun.\n"
            )
        else:
            extra = ""

        prompt = f"""You help a retrieval system search a German food database ({label}).
Given a user food text, output a SHORT rewritten German query using common food words likely in the database.
Rules:
- Keep it under 8 words.
- Prefer generic food nouns over brands.
- If term is "skyr", rewrite to "joghurt eiweissreich" or "quark eiweissreich".
{extra}
Return ONLY the rewritten query.

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
        if not self.llm:
            best = candidates[0]
            # hybrid fallback confidence based on score strength
            s1, margin, is_strong, is_weak = self._candidate_strength(candidates)
            conf = 80 if is_strong else (60 if is_weak else 70)
            return best["code"], best["name"], conf

        cfg = self.ds[dataset_key]["cfg"]
        label = "BLS 3.02" if cfg.key == "bls302" else "BLS 4.0"

        cand_text = "\n".join(
            [f"{i}. {c['code']} - {c['name']}" for i, c in enumerate(candidates[: self.LLM_TOP_N], 1)]
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

            for c in candidates:
                if c["code"] == chosen:
                    # confidence depends on retrieval strength
                    s1, margin, is_strong, is_weak = self._candidate_strength(candidates)
                    if is_strong:
                        return c["code"], c["name"], 92
                    if is_weak:
                        return c["code"], c["name"], 70
                    return c["code"], c["name"], 82

            best = candidates[0]
            s1, margin, is_strong, is_weak = self._candidate_strength(candidates)
            conf = 88 if is_strong else (68 if is_weak else 78)
            return best["code"], best["name"], conf

        except Exception:
            best = candidates[0]
            s1, margin, is_strong, is_weak = self._candidate_strength(candidates)
            conf = 85 if is_strong else (60 if is_weak else 72)
            return best["code"], best["name"], conf

    # ----------------------------
    # Main matching logic (FIXED to match your requirements)
    # ----------------------------
    def match_one(self, dataset_key: str, food_text: str) -> dict:
        raw = str(food_text).strip()

        # (1) Authoritative human lookup
        lh = self._lookup_hit(dataset_key, raw)
        if lh is not None:
            src, code, name = lh
            # still provide candidates for UI debugging (optional)
            cands = self.generate_candidates(dataset_key, raw, top_k=self.TOP_K_RETURN)
            return {
                "food": raw,
                "code": code,
                "name": name,
                "confidence": 100,
                "source": "lookup",
                "match_source": src,
                "rewritten_query": None,
                "candidates": cands[: self.UI_TOP_K],
            }

        # (2) Ambiguous lookup path (if you have it)
        amb = self._ambiguous_candidates_from_lookup(dataset_key, raw)
        if amb:
            # Merge ambiguous list with retrieval list, then LLM choose
            retr0 = self.generate_candidates(dataset_key, raw, top_k=self.TOP_K_RETURN)
            merged = []
            seen = set()
            for c in (amb + retr0):
                if c["code"] in seen:
                    continue
                seen.add(c["code"])
                merged.append(c)

            code, name, conf = self.ask_llm_choose(dataset_key, raw, merged)
            return {
                "food": raw,
                "code": code,
                "name": name,
                "confidence": conf,
                "source": "llm" if self.llm else "hybrid",
                "match_source": "ambiguous_llm" if self.llm else "ambiguous_hybrid",
                "rewritten_query": None,
                "candidates": merged[: self.UI_TOP_K],
            }

        # (3) Normal: retrieval first (NO rewrite yet)
        candidates0 = self.generate_candidates(dataset_key, raw, top_k=self.TOP_K_RETURN)
        s1, margin, is_strong, is_weak = self._candidate_strength(candidates0)

        # If retrieval is strong -> just let LLM pick among top-N (no rewrite)
        if candidates0 and (is_strong or not self.llm):
            code, name, conf = self.ask_llm_choose(dataset_key, raw, candidates0[: self.LLM_TOP_N])
            return {
                "food": raw,
                "code": code,
                "name": name,
                "confidence": conf,
                "source": "llm" if self.llm else "hybrid",
                "match_source": "retrieval_then_choose",
                "rewritten_query": None,
                "candidates": candidates0[: self.UI_TOP_K],
            }

        # If retrieval weak/uncertain and LLM available -> rewrite + retrieve + choose
        rewritten1 = self.llm_rewrite_query(dataset_key, raw, mode="rewrite") if self.llm else raw
        candidates1 = self.generate_candidates(dataset_key, rewritten1, top_k=self.TOP_K_RETURN)
        s1b, marginb, is_strong_b, is_weak_b = self._candidate_strength(candidates1)

        # If still weak, do a rescue rewrite
        rewritten2 = None
        candidates2 = []
        if self.llm and (not candidates1 or is_weak_b):
            rewritten2 = self.llm_rewrite_query(dataset_key, raw, mode="rescue")
            candidates2 = self.generate_candidates(dataset_key, rewritten2, top_k=self.TOP_K_RETURN)

        # Choose best candidate set
        final_rewritten = rewritten1 if (rewritten1 != raw) else None
        final_candidates = candidates1 if candidates1 else candidates0

        if candidates2:
            # take candidates2 if it improved top score
            s2, _, _, _ = self._candidate_strength(candidates2)
            if s2 > s1b:
                final_candidates = candidates2
                final_rewritten = rewritten2 if (rewritten2 and rewritten2 != raw) else final_rewritten

        if not final_candidates:
            return {
                "food": raw,
                "code": "",
                "name": "",
                "confidence": 0,
                "source": "none",
                "match_source": "none",
                "rewritten_query": final_rewritten,
                "candidates": [],
            }

        code, name, conf = self.ask_llm_choose(dataset_key, raw, final_candidates[: self.LLM_TOP_N])
        return {
            "food": raw,
            "code": code,
            "name": name,
            "confidence": conf,
            "source": "llm" if self.llm else "hybrid",
            "match_source": "rewrite_then_choose" if final_rewritten else "retrieval_then_choose",
            "rewritten_query": final_rewritten,
            "candidates": final_candidates[: self.UI_TOP_K],
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
