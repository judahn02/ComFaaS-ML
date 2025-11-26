# pipeline/prune_stage.py

# stdlib
import time

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
# (C에서 M을 쓴다면)
import Metrics as M

# project-local
from Pruning import PruneCfg, SortPruner
from Data import attach_realized_latency

# 공통 유틸 (values_for_sub 등을 쓰는 경우가 많음)
from .common import log, _values_for_sub

# === C) Prune + Realized + Top-K recall =====================================
def _hyb_prune_and_realize(clean: pd.DataFrame,
                           mean: np.ndarray,
                           p95: np.ndarray,
                           uncert: np.ndarray,
                           args) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, Dict]:
    """[C) Prune + Realized + Top-K recall]
    5) SortPruner.select → 로그/kept/kpr → realized 부착 → 6) Top-K recall 까지.
    in : clean, mean, p95, uncert, args
    out: (pruned, realized, recall_pct, t_prune_sec, extra_dict)
    """
    prune_cfg = PruneCfg(
        Kmax=args.K,
        lambda_tail=args.lambda_tail,
        risk_rho=getattr(args, "risk_rho", 0.0),
        risk_omem=getattr(args, "risk_omem", 0.0),
        alpha_u=args.alpha_uncert,
        beta_tail=args.beta_tail,
        buffer_ms=args.buffer_ms,
        m_guard=args.m_guard,
        c_guard=args.c_guard
    )

    t0 = time.time()
    pruned = SortPruner().select(clean, mean, p95, uncert, prune_cfg)
    t_prune = time.time() - t0

    kept = int(len(pruned))
    total = int(len(clean))
    reqs = int(clean['req_id'].nunique())
    kpr = kept / max(reqs, 1)
    log(f"[Prune] kept={kept}/{total} (~{kpr:.0f}/req) | Wall time: {t_prune:.2f} s")

    realized = attach_realized_latency(clean, seed=args.seed + 13)

    # 6) Metrics: Top-K recall (for reference) -------------------------------
    recall = M.topk_oracle_recall(pruned, realized)    
    log(f"[Top-K recall] {recall:.2f}%")

    # ensure local extra dict
    extra = {}

    return pruned, realized, recall, t_prune, extra
