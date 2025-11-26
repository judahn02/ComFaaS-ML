# pipeline/cand_builder.py

import os, time

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


# 기존 프로젝트의 로깅/저장 유틸은 Print 모듈에 있음
from Print import log, save_df, save_json

from .common import _force_determinism

# === G) Build candidate table ===============================================
def _hyb_build_cand(sub: pd.DataFrame,
                    mean: np.ndarray,
                    p95: np.ndarray,
                    scores: np.ndarray,
                    score_src: str,
                    decision_src: str,
                    args) -> pd.DataFrame:
    """[G) Build candidate table] est_total/p95_total/penalties/edge_bias/robustify/dump.
    in : sub, mean, p95, scores, score_src, decision_src, args
    out: cand_df
    """
    # build candidate table for selection
    est_total_ms = (
        sub['est_transfer_ms'].values + sub['est_jitter_ms'].values
        + mean[sub.index] + sub['p_cold_est'].values * sub['est_cold_ms'].values
    )
    p95_total_ms = (
        sub['est_transfer_ms'].values + sub['est_jitter_ms'].values
        + p95[sub.index] + sub['p_cold_est'].values * sub['est_cold_ms'].values
    )
    cand = sub[['req_id','node_id','is_edge','rho_proxy','over_mem_pos']].copy()
    cand['est_total_ms'] = est_total_ms
    cand['p95_total_ms'] = p95_total_ms
    cand['score'] = scores
    cand['score_src'] = score_src
    cand['decision_src'] = decision_src

    # --- p95-based penalties & score (hybrid-heur only) ---
    # 기본 패널티: edge bias(+), rho 초과(−)
    rho_thr = float(getattr(args, 'rho_thr', 0.90))
    rho_penalty = float(getattr(args, 'rho_penalty', 2.0))
    edge_bias = float(getattr(args, 'edge_bias', 0.0))

    _force_determinism(int(getattr(args, "seed", 0)) + 19)

    rho_excess = np.maximum(0.0, cand.get('rho_proxy', 0.0) - rho_thr)
    penalty_rho = rho_penalty * rho_excess
    bonus_edge = edge_bias * cand.get('is_edge', 0.0).astype(float)

    # 최종 p95 기반 점수 (tail 우선 억제)
    cand['score_p95'] = -(cand['p95_total_ms']) + bonus_edge - penalty_rho

    # --- robustify cand columns: map alternative column names if needed ---
    if 'rho_proxy' not in cand.columns:
        if 'rho' in sub.columns:
            cand['rho_proxy'] = sub['rho'].to_numpy(dtype=float)
        elif 'rho_proxy' in sub.columns:
            cand['rho_proxy'] = sub['rho_proxy'].to_numpy(dtype=float)
        else:
            cand['rho_proxy'] = 0.0

    if 'over_mem_pos' not in cand.columns:
        alt_names = [n for n in ['overmem_proxy', 'overmem_pos', 'over_mem', 'overmem'] if n in sub.columns]
        if alt_names:
            cand['over_mem_pos'] = sub[alt_names[0]].to_numpy(dtype=float)
        else:
            cand['over_mem_pos'] = 0.0

    if getattr(args, "dump_cand", False):
        save_df(cand, os.path.join(args.outdir, f'cand_{decision_src}.csv'))

    return cand
