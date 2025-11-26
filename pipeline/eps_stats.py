# pipeline/eps_stats.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Any, Dict

from .common import _values_for_sub  # mean을 index 순서로 뽑는 헬퍼


# === J) ε-trigger stats ======================================================
def _hyb_eps_stats(clean: pd.DataFrame,
                   cand: pd.DataFrame,
                   mean: np.ndarray,
                   args) -> Tuple[float, float]:
    """
    ε-trigger rates (full vs post-prune).
    정의: 각 req에서 est_total 오름차순 1등과 2등의 차이가 eps 이하이면 trigger.
    반환: (eps_full_pct, eps_post_pct) 둘 다 퍼센트(%) 스케일의 float.
    구현: 완전 벡터화(세그먼트 연산), 불필요한 copy 금지.
    """
    eps = float(args.eps)

    def _percent(df: pd.DataFrame, mean_vec: np.ndarray) -> float:
        if df.empty:
            return 0.0

        # df.index에 맞춘 mean 직접 결합 (불필요 copy/realign 방지)
        idx = df.index.to_numpy()
        # est_total_ms = transfer + jitter + mean + p_cold*est_cold (존재하는 컬럼만 합산)
        est = mean_vec[idx].astype(np.float64)
        if 'est_transfer_ms' in df.columns:
            est = est + df['est_transfer_ms'].to_numpy(np.float64, copy=False)
        if 'est_jitter_ms' in df.columns:
            est = est + df['est_jitter_ms'].to_numpy(np.float64, copy=False)
        if 'p_cold_est' in df.columns and 'est_cold_ms' in df.columns:
            est = est + (df['p_cold_est'].to_numpy(np.float64, copy=False) *
                         df['est_cold_ms'].to_numpy(np.float64, copy=False))

        req = df['req_id'].to_numpy(np.int64, copy=False)

        # req별 + est 오름차순 정렬 (세그먼트 계산을 위한 전처리)
        order = np.lexsort((est, req))
        req_o = req[order]
        est_o = est[order]

        # 세그먼트 경계 (req 변화 지점)
        cut = np.r_[True, req_o[1:] != req_o[:-1]]
        seg_st = np.flatnonzero(cut)
        seg_en = np.r_[seg_st[1:], req_o.size]  # [start, end)

        # 각 req에서 후보 수(n)와 1등/2등 차 계산
        total = 0
        trig  = 0
        # 루프는 "세그먼트 수(=req 수)"에만 비례 → 매우 작음/빠름
        for s, e in zip(seg_st, seg_en):
            n = e - s
            if n >= 2:
                total += 1
                # est_o는 이미 오름차순이므로 s가 1등, s+1이 2등
                if (est_o[s + 1] - est_o[s]) <= eps:
                    trig += 1

        return (trig / total * 100.0) if total else 0.0

    eps_full_pct = _percent(clean, mean)
    eps_post_pct = _percent(cand,  mean)
    
    return float(eps_full_pct), float(eps_post_pct)

# # === J) ε-trigger stats ======================================================
# def _hyb_eps_stats(clean: pd.DataFrame,
#                    cand: pd.DataFrame,
#                    mean: np.ndarray,
#                    args) -> tuple[float, float]:
#     """
#     ε-trigger rates (full vs post-prune)를 벡터화로 계산.
#     정의: req별 est_total에서 '베스트와 eps 이내에 드는 후보 수 >= 2' 이면 트리거 발생.
#     반환: (eps_full_pct, eps_post_pct)  # 둘 다 %
#     """
#     eps = float(getattr(args, "eps", 0.0))

#     def _rate(df: pd.DataFrame, need_recompute_total: bool) -> float:
#         if df.empty:
#             return 0.0

#         if need_recompute_total:
#             mu = _values_for_sub(mean, df.index).reshape(-1).astype(np.float64)
#             est_transfer = df['est_transfer_ms'].to_numpy(float) if 'est_transfer_ms' in df.columns else 0.0
#             est_jitter   = df['est_jitter_ms'].to_numpy(float)   if 'est_jitter_ms'   in df.columns else 0.0
#             cold_term    = (
#                 df['p_cold_est'].to_numpy(float) * df['est_cold_ms'].to_numpy(float)
#                 if 'p_cold_est' in df.columns and 'est_cold_ms' in df.columns else 0.0
#             )
#             est = est_transfer + est_jitter + mu + cold_term
#         else:
#             # cand 에 이미 est_total_ms가 있다고 가정
#             est = df['est_total_ms'].to_numpy(float)

#         s = pd.Series(est, index=df.index)
#         best = s.groupby(df['req_id']).transform('min')         # 각 req의 최소(est_total)
#         near = (s - best) <= eps                                # eps 윈도우 내 여부 (베스트 포함)
#         # 윈도우 내 개수
#         near_count = near.groupby(df['req_id']).transform('sum')
#         # 개수 >= 2 → 트리거
#         triggered = (near_count >= 2).groupby(df['req_id']).any()
#         return float(triggered.mean() * 100.0)

#     # clean은 est_total 재계산, cand는 이미 est_total_ms 있음(없으면 True로 바꾸면 됨)
#     eps_full_pct = _rate(clean, need_recompute_total=True)
#     eps_post_pct = _rate(cand,  need_recompute_total=('est_total_ms' not in cand.columns))

#     return eps_full_pct, eps_post_pct

