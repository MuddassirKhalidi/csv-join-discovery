# enrich.py
import os, json, itertools, math
import numpy as np
import pandas as pd

# ---------- Utilities

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_id_like(name: str, s: pd.Series) -> bool:
    n = name.lower()
    if any(tok in n for tok in ["_id", "uuid", "guid", "hash", "index", "code"]):
        # if it's integer-but-unique, likely an identifier
        if _is_numeric(s):
            return s.nunique(dropna=True) > 0.9 * len(s)
        return True
    return False

def _missing_rate(s: pd.Series) -> float:
    return float(s.isna().mean())

def _coverage(s: pd.Series) -> float:
    return float(s.notna().mean())

def _max_abs_corr(cand: pd.Series, base_numeric_df: pd.DataFrame) -> float:
    x = pd.to_numeric(cand, errors="coerce")
    best = 0.0
    for col in base_numeric_df.columns:
        y = pd.to_numeric(base_numeric_df[col], errors="coerce")
        df = pd.concat([x, y], axis=1).dropna()
        if len(df) < 5:
            continue
        r = abs(df.iloc[:,0].corr(df.iloc[:,1]))
        if pd.notna(r):
            best = max(best, float(r))
    return best

def _plausible(col_name: str, s: pd.Series) -> bool:
    x = pd.to_numeric(s, errors="coerce")
    frac = x.notna().mean()
    if frac < 0.5:  # too sparse to judge
        return False
    n = col_name.lower()
    # generic sanity: finite and not absurd outliers
    x = x[np.isfinite(x)]
    if x.empty:
        return False
    q1, q99 = np.quantile(x, [0.01, 0.99])
    if not np.isfinite(q1) or not np.isfinite(q99):
        return False
    # rate/percentage fields
    if any(tok in n for tok in ["rate", "pct", "%", "percentage"]):
        return (x.min() >= -1) and (x.max() <= 100.5)
    # counts/measures should not be wildly negative
    if any(tok in n for tok in ["gdp","revenue","pop","count","cases","deaths","sales","income"]):
        return x.max() <= 1e15 and x.min() >= -1e5
    return True

def _relevance_score(col_name: str) -> float:
    # Lightweight name-based relevance; tune tokens per domain
    name = col_name.lower()
    POS = ["gdp","inflation","mortality","fertility","life_expectancy",
           "internet","electricity","education","literacy","unemployment",
           "growth","co2","emission","poverty","access","health","income","exports","imports"]
    NEG = ["note","comment","description","footnote","source"]
    pos = sum(tok in name for tok in POS)
    neg = sum(tok in name for tok in NEG)
    return max(0.0, min(1.0, 0.2*pos - 0.2*neg + (0.2 if any(c in name for c in ["rate","per_capita","index"]) else 0.0)))

def _base_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    num = {c for c in df.columns if _is_numeric(df[c])}
    # also allow numeric-castable columns with decent success rate
    for c in df.columns:
        if c in num: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.8:
            num.add(c)
    return df[list(num)] if num else pd.DataFrame(index=df.index)

# ---------- Join-key discovery

def _value_overlap_ratio(a: pd.Series, b: pd.Series) -> float:
    A = set(pd.Series(a).dropna().unique())
    B = set(pd.Series(b).dropna().unique())
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = min(len(A), len(B))
    return inter / denom if denom else 0.0

def _is_unique_combo(df: pd.DataFrame, cols: list) -> bool:
    if not cols: return False
    return df[cols].dropna().duplicated().sum() == 0

def find_join_keys(base: pd.DataFrame, other: pd.DataFrame,
                   max_key_len: int = 2, min_overlap: float = 0.8):
    """
    Returns list of tuples [(base_cols, other_cols)] of length 1 or 2
    that look like viable join keys.
    """
    candidates = []
    # prefer exact same column names first
    shared = [c for c in base.columns if c in other.columns]
    for c in shared:
        ov = _value_overlap_ratio(base[c], other[c])
        if ov >= min_overlap:
            candidates.append(([c], [c], ov))

    # cross-name single-column matches
    for bcol in base.columns:
        for ocol in other.columns:
            if bcol == ocol: continue
            ov = _value_overlap_ratio(base[bcol], other[ocol])
            if ov >= min_overlap:
                candidates.append(([bcol], [ocol], ov))

    # composite keys up to length 2
    bcols = list(base.columns)
    ocols = list(other.columns)
    for k in range(2, max_key_len+1):
        for bpair in itertools.combinations(bcols, k):
            for opair in itertools.combinations(ocols, k):
                # quick screen: each individual column should have some overlap
                if any(_value_overlap_ratio(base[bc], other[oc]) < min_overlap*0.6
                       for bc, oc in zip(bpair, opair)):
                    continue
                # build tuple keys
                bkey = base[list(bpair)].astype(str).agg("||".join, axis=1)
                okey = other[list(opair)].astype(str).agg("||".join, axis=1)
                ov = _value_overlap_ratio(bkey, okey)
                if ov >= min_overlap:
                    candidates.append((list(bpair), list(opair), ov))

    # rank by overlap, prefer combos that are unique in both
    def score(item):
        bks, oks, ov = item
        s = ov
        if _is_unique_combo(base, bks): s += 0.05
        if _is_unique_combo(other, oks): s += 0.05
        return s

    candidates.sort(key=score, reverse=True)
    # return as list of dicts
    return [{"base_keys": b, "other_keys": o, "overlap": float(ov)} for b,o,ov in candidates]

# ---------- Column selection per file

def select_enrichment_columns(base: pd.DataFrame,
                              other: pd.DataFrame,
                              base_keys: list,
                              other_keys: list,
                              cov_thresh=0.70, miss_thresh=0.30,
                              rel_thresh=0.60, nov_thresh=0.10):
    """
    Returns [{'col': name, 'score': float}] for columns to add from other.
    """
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
        if not _plausible(c, s): continue
        miss = _missing_rate(s)
        if miss > miss_thresh: continue
        rel = _relevance_score(c)
        if rel < rel_thresh: continue
        nov = 1.0 - _max_abs_corr(s, base_num) if not base_num.empty else 1.0
        if nov < nov_thresh: continue
        score = 0.4*cov + 0.3*rel + 0.3*nov
        picks.append({"col": c, "score": float(score)})
    picks.sort(key=lambda x: x["score"], reverse=True)
    return picks

# ---------- End-to-end enrichment

def enrich_directory(base_csv: str, dir_path: str, out_csv="enriched.csv", out_log="provenance.json",
                     explicit_base_keys: list | None = None, top_k_global: int = 50, min_overlap=0.8):
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
        # find keys
        if explicit_base_keys is not None:
            # map base->other by same names or first match
            keys_map = [{"base_keys": explicit_base_keys,
                         "other_keys": [k for k in explicit_base_keys if k in df.columns],
                         "overlap": 0.0}] if all(k in df.columns for k in explicit_base_keys) else []
        else:
            keys_map = find_join_keys(base, df, max_key_len=2, min_overlap=min_overlap)

        if not keys_map:
            continue
        bk = keys_map[0]["base_keys"]
        ok = keys_map[0]["other_keys"]

        picks = select_enrichment_columns(base, df, bk, ok)
        for p in picks:
            selections.append({"file": f, "base_keys": bk, "other_keys": ok,
                               "col": p["col"], "score": p["score"]})

    # keep top-K globally
    selections.sort(key=lambda x: x["score"], reverse=True)
    keep = selections[:top_k_global]

    # build enriched frame
    enriched = base.copy()
    for s in keep:
        df = pd.read_csv(s["file"], usecols=list(set(s["other_keys"] + [s["col"]])))
        slim = df.rename(columns={ok: bk for ok, bk in zip(s["other_keys"], s["base_keys"])})
        suffix = "__" + os.path.splitext(os.path.basename(s["file"]))[0]
        to_merge = slim[s["base_keys"] + [s["col"]]]
        enriched = pd.merge(enriched, to_merge, on=s["base_keys"], how="left", validate="m:1", copy=False, suffixes=("", suffix))
        # ensure unique name with suffix when collisions happen
        if s["col"] in base.columns:
            new_name = f"{s['col']}{suffix}"
            enriched.rename(columns={s["col"]: new_name}, inplace=True)
            s["renamed_to"] = new_name

    enriched.to_csv(out_csv, index=False)
    with open(out_log, "w") as f:
        json.dump(keep, f, indent=2)
    return out_csv, out_log
