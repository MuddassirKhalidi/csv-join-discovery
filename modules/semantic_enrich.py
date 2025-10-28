# -------- Semantic helpers --------
import re
import os
import json
import itertools
import numpy as np
import pandas as pd

# Import needed functions from naive_enrich (same module package)
from .naive_enrich import (
    find_join_keys, _value_overlap_ratio, _is_numeric, 
    _is_id_like, _missing_rate, _coverage, _base_numeric_cols, _max_abs_corr
)

_UNIT_PATTERNS = {
    "percent": re.compile(r"(percent|percentage|%|\b_pct\b|\brate\b)", re.I),
    "per_1000": re.compile(r"per\s*1000|\b/1000\b", re.I),
    "usd": re.compile(r"\b(usd|us\$|\$)\b|\b(ppp)\b|\bcurrency\b", re.I),
    "index": re.compile(r"\bindex\b|\bscore\b|\brating\b", re.I),
    "per_capita": re.compile(r"per[_\s]?capita|\bpc\b", re.I),
}

def _detect_unit_hint(col_name: str) -> set[str]:
    hits = set()
    for k, pat in _UNIT_PATTERNS.items():
        if pat.search(col_name or ""):
            hits.add(k)
    return hits

def _sample_values(series, k=8):
    s = series.dropna()
    if len(s) == 0: return []
    return list(s.sample(min(k, len(s)), random_state=42).astype(str))

def _num_stats(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) == 0:
        return {"is_numeric": False}
    return {
        "is_numeric": True,
        "n": len(x),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "uniq_ratio": float(pd.Series(x).nunique() / max(1, len(x))),
    }

# -------- 1) Semantic PLAUSIBILITY --------
def semantic_plausibility(col_name: str, series: pd.Series) -> float:
    """
    Returns a plausibility score in [0,1] by checking if the column's
    NAME/UNITS agree with VALUE SHAPE (range, sign, variability).
    """
    name = (col_name or "").lower()
    unit_hints = _detect_unit_hint(name)
    st = _num_stats(series)

    # if non-numeric column, plausibility is low for measures
    if not st.get("is_numeric", False):
        return 0.0

    score = 0.0
    checks = 0

    # 1) Range vs unit hint
    checks += 1
    if "percent" in unit_hints:
        # allow 0..100 (or 0..1 if given as fraction)
        if (st["min"] >= -1 and st["max"] <= 100.5):
            score += 1.0
    elif "per_1000" in unit_hints:
        checks += 0  # keep same weight
        if (st["min"] >= 0 and st["max"] <= 2000):
            score += 1.0
        checks += 1
    elif "usd" in unit_hints:
        checks += 1
        if st["max"] < 1e15 and st["min"] >= -1e5:
            score += 1.0
    elif "index" in unit_hints:
        checks += 1
        if (st["max"] <= 10000 and st["min"] >= -100):  # generous
            score += 1.0
    else:
        checks += 1
        # generic sanity
        if st["max"] <= 1e15 and st["min"] >= -1e6:
            score += 1.0

    # 2) Not an ID: avoid near-unique integers disguised as measures
    checks += 1
    if st["uniq_ratio"] < 0.95 or "index" in unit_hints:  # indices may be high-unique
        score += 1.0

    # 3) Variability exists
    checks += 1
    if st["std"] > 0:  # non-constant
        score += 1.0
    # normalize to [0,1]
    return float(score / checks)

# -------- 2) Semantic RELEVANCE --------
# Uses embeddings if available; falls back to TF-IDF cosine.
def _cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _tfidf_cosine(query: str, text: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        # ultra-simple fallback: token overlap
        qa = set(re.findall(r"\w+", (query or "").lower()))
        ta = set(re.findall(r"\w+", (text or "").lower()))
        return len(qa & ta) / max(1, len(qa))
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform([query or "", text or ""])
    return float(cosine_similarity(X[0], X[1])[0,0])

def _embed_texts(texts: list[str]):
    try:
        # Optional SBERT path; user can install `sentence-transformers`
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, normalize_embeddings=True)
    except Exception:
        return None  # fall back to TF-IDF

def semantic_relevance(query: str, col_name: str, series: pd.Series, context: str | None = None) -> float:
    """
    Returns a relevance score in [0,1] between a user/domain query and a column.
    Representation = header + few samples + optional context (filename/page title).
    """
    samples = _sample_values(series, k=6)
    text = f"{col_name} | samples: {', '.join(samples)} | context: {context or ''}"

    # Try embeddings first
    vecs = _embed_texts([query or "", text])
    if vecs is not None:
        sim = _cosine(vecs[0], vecs[1])
        # map [-1,1]→[0,1] though cosine from normalized SBERT is already [0,1]
        return float(max(0.0, min(1.0, sim)))

    # Fallback: TF-IDF cosine
    return float(max(0.0, min(1.0, _tfidf_cosine(query, text))))


# -------- 3) Semantic Column Selection --------
def select_enrichment_columns_semantic(base: pd.DataFrame,
                                       other: pd.DataFrame,
                                       base_keys: list,
                                       other_keys: list,
                                       cov_thresh=0.70, 
                                       nov_thresh=0.10):
    """
    Semantic version of column selection.
    Uses semantic_relevance + semantic_plausibility instead of naive checks.
    Query = column names from base table (excluding join keys).
    Returns [{'col': name, 'score': float}] for columns to add from other.
    """
    # Build query from base table column names (excluding join keys)
    base_cols = [c for c in base.columns if c not in base_keys]
    query = " ".join(base_cols)
    
    # Align by keys
    joined = pd.merge(
        base[base_keys].drop_duplicates(),
        other[other_keys + [c for c in other.columns if c not in other_keys]],
        left_on=base_keys, right_on=other_keys, how="left",
        copy=False, validate="m:1"
    )
    
    # base numeric matrix for novelty check
    base_num = _base_numeric_cols(base.drop(columns=base_keys, errors="ignore"))
    
    picks = []
    for c in other.columns:
        if c in other_keys: continue
        s = joined[c]
        if not _is_numeric(s):
            # allow mostly-numeric strings
            s = pd.to_numeric(s, errors="coerce")
        
        cov = _coverage(s)
        if cov < cov_thresh: continue
        if _is_id_like(c, other[c]): continue
        
        # SEMANTIC CHECKS (replacing naive ones)
        plaus = semantic_plausibility(c, s)
        if plaus < 0.3: continue
        
        # Context = filename or table identifier
        context = None  # Could be passed from caller
        
        rel = semantic_relevance(query, c, s, context)
        if rel < 0.1: continue  # minimal semantic relevance threshold
        
        nov = 1.0 - _max_abs_corr(s, base_num) if not base_num.empty else 1.0
        if nov < nov_thresh: continue
        
        # Weighted score (emphasize semantic relevance and plausibility)
        score = 0.4*cov + 0.3*rel + 0.2*plaus + 0.1*nov
        picks.append({"col": c, "score": float(score)})
    
    picks.sort(key=lambda x: x["score"], reverse=True)
    return picks


# -------- 4) End-to-end Semantic Enrichment --------
def enrich_directory_semantic(base_csv: str, dir_path: str, out_csv="enriched_semantic.csv", 
                              out_log="provenance_semantic.json",
                              explicit_base_keys: list | None = None, 
                              top_k_global: int = 50, 
                              min_overlap=0.8):
    """
    Semantic version of directory enrichment.
    Uses semantic relevance and plausibility for column selection.
    """
    base = pd.read_csv(base_csv)
    
    # discover CSVs
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith(".csv") and os.path.join(dir_path, f) != base_csv]
    
    selections = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        
        # find keys (same as naive)
        if explicit_base_keys is not None:
            keys_map = [{"base_keys": explicit_base_keys,
                         "other_keys": [k for k in explicit_base_keys if k in df.columns],
                         "overlap": 0.0}] if all(k in df.columns for k in explicit_base_keys) else []
        else:
            keys_map = find_join_keys(base, df, max_key_len=2, min_overlap=min_overlap)
        
        if not keys_map:
            continue
        
        bk = keys_map[0]["base_keys"]
        ok = keys_map[0]["other_keys"]
        
        # Use semantic column selection instead of naive
        picks = select_enrichment_columns_semantic(base, df, bk, ok)
        
        for p in picks:
            selections.append({"file": f, "base_keys": bk, "other_keys": ok,
                               "col": p["col"], "score": p["score"]})
    
    # keep top-K globally
    selections.sort(key=lambda x: x["score"], reverse=True)
    keep = selections[:top_k_global]
    
    # build enriched frame (same as naive)
    enriched = base.copy()
    for s in keep:
        df = pd.read_csv(s["file"], usecols=list(set(s["other_keys"] + [s["col"]])))
        slim = df.rename(columns={ok: bk for ok, bk in zip(s["other_keys"], s["base_keys"])})
        suffix = "__" + os.path.splitext(os.path.basename(s["file"]))[0]
        to_merge = slim[s["base_keys"] + [s["col"]]]
        enriched = pd.merge(enriched, to_merge, on=s["base_keys"], how="left", 
                           validate="m:1", copy=False, suffixes=("", suffix))
        # ensure unique name with suffix when collisions happen
        if s["col"] in base.columns:
            new_name = f"{s['col']}{suffix}"
            enriched.rename(columns={s["col"]: new_name}, inplace=True)
            s["renamed_to"] = new_name
    
    enriched.to_csv(out_csv, index=False)
    with open(out_log, "w") as f:
        json.dump(keep, f, indent=2)
    return out_csv, out_log
