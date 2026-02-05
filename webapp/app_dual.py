import os
from pathlib import Path
import streamlit as st

from matcher_multi import DualFoodMatcher

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # webapp/.. = project root
PROJECT_ROOT = Path(os.getenv("BLS_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).expanduser().resolve()

st.set_page_config(page_title="BLS Dual Matcher", layout="wide")
st.title("BLS 3.02 + BLS 4.0 Food Matcher")

@st.cache_resource
def load_matcher(project_root: Path):
    return DualFoodMatcher(project_root)

m = load_matcher(PROJECT_ROOT)

def sort_candidates(cands):
    def to_float(x):
        try:
            return float(x.get("score", 0))
        except Exception:
            return 0.0
    return sorted(cands, key=to_float, reverse=True)

with st.sidebar:
    st.subheader("Status")
    st.write(m.get_stats())

    if st.button("🔄 Reload matcher (re-read lookups)"):
        st.cache_resource.clear()
        st.rerun()

    show_candidates = st.checkbox("Show top candidates", value=True)
    topn = st.slider("How many candidates", 5, 50, 10)
    show_candidates_for_lookup = st.checkbox("Show candidates even for lookup hits", value=True)

food = st.text_input("Food item", placeholder="e.g., Alpro Haferdrink ungesüßt 1l")

with st.form("search_form", clear_on_submit=False):
    submitted = st.form_submit_button("Search", type="primary")

if submitted and food.strip():
    out = m.match_both(food)

    colA, colB = st.columns(2)

    def render_panel(col, label, dataset_key, result):
        with col:
            st.subheader(label)
            st.write("**Source:**", result.get("source", ""))
            st.write("**Match source:**", result.get("match_source", ""))
            st.write("**Confidence:**", result.get("confidence", ""))
            st.write("**Code:**", result.get("code", ""))
            st.write("**Name:**", result.get("name", ""))

            if result.get("rewritten_query"):
                st.caption(f"LLM rewrite: {result['rewritten_query']}")

            st.divider()
            st.subheader("Confirm & Learn")
            if result.get("code"):
                if st.button(f"✅ Confirm {label} and save", key=f"confirm_{dataset_key}"):
                    m.add_to_lookup(dataset_key, food, result["code"])
                    st.success("Saved (lookup_user updated).")

            if show_candidates:
                st.divider()
                st.subheader("Top Candidates")

                cands = result.get("candidates", []) or []

                if (result.get("source") == "lookup") and show_candidates_for_lookup:
                    try:
                        cands = m.generate_candidates(dataset_key, food, top_k=max(50, topn))
                    except Exception:
                        pass

                cands = sort_candidates(cands)
                for i, c in enumerate(cands[:topn], 1):
                    st.write(f"{i}. **{c.get('code','')}** | {c.get('name','')} | score={c.get('score','')}")

    render_panel(colA, "BLS 3.02", "bls302", out["bls302"])
    render_panel(colB, "BLS 4.0", "bls40", out["bls40"])
