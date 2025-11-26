# pipeline/scoring_ops.py

# third-party
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


from Reranker import ListNetReranker

# project-local (공통 유틸)
from .common import log, _values_for_sub

# (F에서 리랭커 특성 빌더를 쓰는 경우)
from Features import RerankerBlock  # _hyb_score_ml 내부에서 사용한다면 필요

# (F에서 torch no_grad()를 쓰는 경우)
import torch  # _hyb_score_ml 내부에서 with torch.no_grad(): 등을 사용한다면 필요


# === [module-level helpers] ================================================

def _robust_z(x: "np.ndarray", eps: float = 1e-6) -> "np.ndarray":
    """
    Robust Z-score using median/MAD. Safe for heavy-tailed features.
    Returns float array same shape as x.
    """
    import numpy as _np
    x = _np.asarray(x, dtype=float)
    if x.size == 0:
        return x.astype(float)
    med = _np.median(x)
    mad = _np.median(_np.abs(x - med))
    denom = max(mad * 1.4826, eps)  # MAD -> stdev approx
    return (x - med) / denom


def _quick_corr(a: "np.ndarray", b: "np.ndarray") -> float:
    """
    Fast Pearson correlation (no deps). Returns nan if degenerate.
    """
    import numpy as _np
    a = _np.asarray(a, dtype=float)
    b = _np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    av = a - a.mean()
    bv = b - b.mean()
    denom = (float((av * av).sum()) ** 0.5) * (float((bv * bv).sum()) ** 0.5)
    return float((av * bv).sum() / denom) if denom > 0 else float("nan")


# # === E) Heuristic scoring ====================================================
# def _hyb_score_heur(sub: pd.DataFrame,
#                     mean: np.ndarray,
#                     p95: np.ndarray,
#                     uncert: np.ndarray,
#                     args) -> Tuple[pd.DataFrame, np.ndarray, str, str]:
#     """[E) Heuristic scoring] tail-margin/uncert/qdepth 패널티 포함 휴리스틱 점수 계산.
#     in : pruned, mean, p95, uncert, args
#     out: (sub_df, scores, score_src='heur', decision_src='hybrid')
#     """
#     # 31.1.1 fallback heuristic scoring (tail-margin 포함)
#     _mean = _values_for_sub(mean,   sub.index)
#     _p95  = _values_for_sub(p95,    sub.index)
#     _u    = _values_for_sub(uncert, sub.index)

#     est_total_ = (
#         sub['est_transfer_ms'].to_numpy(dtype=float)
#         + sub['est_jitter_ms'].to_numpy(dtype=float)
#         + _mean
#         + sub['p_cold_est'].to_numpy(dtype=float) * sub['est_cold_ms'].to_numpy(dtype=float)
#     )

#     tail_margin = np.maximum(0.0, _p95 - _mean)
#     lambda_tail = float(getattr(args, "lambda_tail", 0.5))

#     # 후보별 est_total_ 근접도(0~1): 최저 est와의 차이가 delta_ms 이내일수록 1에 가까움
#     est = est_total_
#     delta_ms = float(getattr(args, "qd_delta_ms", 1.0))  # 기본 1ms
#     if est.size > 0:
#         est_min = est.min()
#         # closeness = 1 - (차이 / delta_ms), [0,1]로 클립
#         closeness = np.clip(1.0 - np.maximum(0.0, (est - est_min)) / max(delta_ms, 1e-6), 0.0, 1.0)
#     else:
#         closeness = 0.0

#     # 80.1.2 optional: verbose debug only when --debug (block A)
#     if getattr(args, "debug", False):
#         DBG_TAG = "[A]"  # 이 블록을 식별하는 태그
#         # near-mask share
#         try:
#             log(f"[debug]{DBG_TAG} near-mask share = {(closeness > 0).mean():.3f} (delta_ms={delta_ms})")
#         except Exception:
#             log(f"[debug]{DBG_TAG} near-mask share = (unavailable)")
#         # queue_depth 존재 여부
#         try:
#             log(f"[debug]{DBG_TAG} has queue_depth? {'queue_depth' in sub.columns}")
#         except Exception:
#             log(f"[debug]{DBG_TAG} has queue_depth? (unavailable)")

#     qd = sub['queue_depth'].to_numpy(dtype=float) if 'queue_depth' in sub.columns else 0.0
#     # # --- qdepth normalization (std-scale) ---
#     # if not isinstance(qd, float):
#     #     qd_std = qd.std() if qd.size > 1 else 0.0
#     #     if qd_std > 0.0:
#     #         qd = (qd - qd.mean()) / qd_std
#     #         qd = np.clip(qd, -3.0, 3.0)  # guard extreme z-scores
#     # # ---------------------------------------
#     # --- qdepth → ms scale (stronger influence) ---
#     if not isinstance(qd, float):
#         ms_scale = est_total_.mean() if est_total_.size > 0 else 1.0  # ms
#         # 선택 1) 단순 배율: 큐 깊이를 평균 ms 크기로 스케일
#         qd = qd * ms_scale
#         # 선택 2) 표준화 후 ms 배율 (원하면 위 한 줄 대신 아래 3줄 사용)
#         # qd_std = qd.std() if qd.size > 1 else 0.0
#         # if qd_std > 0.0:
#         #     qd = np.clip((qd - qd.mean()) / qd_std, -3.0, 3.0) * ms_scale
#     # ----------------------------------------------
#     scores = (
#         - est_total_
#         - lambda_tail * tail_margin
#         - float(args.uncert_penalty) * _u
#         - float(args.risk_rho2)  * np.maximum(0.0, sub['rho_proxy'].to_numpy(dtype=float) - 0.9)
#         - float(args.risk_omem2) * np.maximum(0.0, sub['over_mem_pos'].to_numpy(dtype=float))
#         # - float(getattr(args, 'qdepth_penalty', 0.0)) * qd
#         - float(getattr(args, 'qdepth_penalty', 0.0)) * qd * closeness
#     )

#     sub['score'] = scores

#     return sub, scores, "heur", "hybrid"

# === E) Heuristic scoring ====================================================
def _hyb_score_heur(sub: pd.DataFrame,
                    mean: np.ndarray,
                    p95: np.ndarray,
                    uncert: np.ndarray,
                    args) -> Tuple[pd.DataFrame, np.ndarray, str, str]:
    """
    [E) Heuristic scoring — EST-first]
    목적: pre-ML 성능(Oracle match (Hybrid-Heur, pre-ML))을 올리기 위해
          '예상 총지연(est_total)'을 직접 최소화하는 점수식을 사용.
    - 전송/지터에 가중치(heur_w_transfer/jitter) 실제 반영
    - tail-margin/uncert/risk를 약하게 패널티
    - qdepth는 '근접 후보(최저 est 근방)'에서만 영향 주도록 closeness 가중
    - sub['score']에 float32로 채움 (계약 준수)
    - 진단 로그: corr(score, -est_total_ms), delta_ms, 사용 가중치
    """

    # 0) 안전 추출
    _mean = _values_for_sub(mean,   sub.index)
    _p95  = _values_for_sub(p95,    sub.index)
    _u    = _values_for_sub(uncert, sub.index)

    # 1) EST total (가중치 반영)
    wT = float(getattr(args, "heur_w_transfer", 1.0))
    wJ = float(getattr(args, "heur_w_jitter",  0.0))

    est_total_ms = (
        wT * sub['est_transfer_ms'].to_numpy(dtype=float)
      + wJ * sub['est_jitter_ms'].to_numpy(dtype=float)
      + _mean
      + sub['p_cold_est'].to_numpy(dtype=float) * sub['est_cold_ms'].to_numpy(dtype=float)
    )

    # 2) tail-margin 및 eps-근접도
    tail_margin  = np.maximum(0.0, _p95 - _mean)                  # p95 - mean
    lambda_tail  = float(getattr(args, "lambda_tail", 0.7))       # 기본 0.7
    delta_ms     = float(getattr(args, "qd_delta_ms", 1.2))       # 근접창(기본 1.2ms 권장)

    if est_total_ms.size > 0:
        est_min   = float(est_total_ms.min())
        closeness = np.clip(1.0 - np.maximum(0.0, (est_total_ms - est_min)) / max(delta_ms, 1e-6),
                            0.0, 1.0)
    else:
        closeness = 0.0

    # 3) qdepth (근접 후보에서만 미세 패널티)
    qdepth_pen = float(getattr(args, "qdepth_penalty", 0.0))
    if 'queue_depth' in sub.columns:
        qd = sub['queue_depth'].to_numpy(dtype=float)
        # ms 스케일로 완만히 => 평균 est 크기를 배율로 사용
        ms_scale = float(est_total_ms.mean()) if est_total_ms.size > 0 else 1.0
        qd_ms = qd * ms_scale
    else:
        qd_ms = 0.0

    # 4) 위험 패널티(약하게)
    wR = float(getattr(args, "heur_w_rho", 0.0))
    wM = float(getattr(args, "heur_w_mem", 0.0))
    rho_over = np.maximum(0.0, sub['rho_proxy'].to_numpy(dtype=float) - 0.9)  # 0.9 넘는 초과만
    mem_over = np.maximum(0.0, sub['over_mem_pos'].to_numpy(dtype=float))      # 양수 초과분만

    # 5) 점수식 (낮은 총지연이 높은 점수)
    #   - est_total_ms 를 직접 최소화 => 부호(-)
    #   - tail-margin/uncert 는 약한 가중으로 페널티
    #   - qdepth는 근접 후보일수록 영향 (closeness 가중)
    #   - rho/mem 리스크는 매우 약하게 가중 (튜너블)
    uncert_pen  = float(getattr(args, "uncert_penalty", 0.0))

    score = (
        - est_total_ms
        - lambda_tail * tail_margin
        - uncert_pen * _u
        - wR * rho_over
        - wM * mem_over
        - (qdepth_pen * (qd_ms * (closeness if isinstance(closeness, np.ndarray) else 1.0)))
    ).astype(np.float32)

    # 6) edge_bias (요청 내부 동점 깨기용 소량 편향)
    edge_bias = float(getattr(args, "edge_bias", 0.0))
    if edge_bias:
        s = float(score.std()) or 1.0
        bump = (((sub["node_id"].to_numpy() % 7) - 3).astype(np.float32)) * (edge_bias * 0.001 * s)
        score = (score + bump).astype(np.float32)

    # 7) 진단 로그 (debug 모드일 때만)
    if getattr(args, "debug", False):
        try:
            # 휴리스틱이 est_total_ms와 "음의 상관"이어야 좋음 => corr(score, -est) < 0
            corr = float(np.corrcoef(score, -est_total_ms)[0, 1])
            print(f"[Heur] wT={wT:.2f} wJ={wJ:.2f} wR={wR:.2f} wM={wM:.2f} "
                  f"| corr(score, -est_ms)={corr:.3f} | delta_ms={delta_ms:.2f}")
        except Exception:
            pass

    # 8) 계약 준수: sub['score'] 채우고 반환
    sub['score'] = score

    return sub, score, "heur", "hybrid"


# === F) ML scoring ===========================================================
def _hyb_score_ml(reranker: "ListNetReranker",
                  sub: pd.DataFrame,
                  mean: np.ndarray,
                  p95: np.ndarray,
                  uncert: np.ndarray,
                  args,
                  mode: str) -> Tuple[pd.DataFrame, np.ndarray, str, str]:
    """[F) ML scoring] reranker.score()로 점수 계산, decision_src는 (hybrid-ml|pure-ml).
    in : reranker, pruned, mean, p95, uncert, args, mode
    out: (sub_df, scores, score_src='ml', decision_src)
    """
    # ADDED: prevent grad/graph creation during inference for determinism
    import torch as _torch
    _torch.set_grad_enabled(False)

    # feats_sub = RerankerBlock().build(sub, mean[sub.index], p95[sub.index], uncert[sub.index])
    use_opsigs = bool(getattr(args, "reranker_use_opsigs", False) and not getattr(args, "reranker_opsigs_off", False))
    feats_sub = RerankerBlock().build(
        sub, mean[sub.index], p95[sub.index], uncert[sub.index],
        mode=mode, use_opsigs=use_opsigs
    )

    with _torch.no_grad():
        scores = reranker.score(feats_sub)

    _torch.set_grad_enabled(True)

    # return sub, scores, "ml", ("hybrid-ml" if mode == "hybrid" else "pure-ml")

    # --- Hybrid-ML only: add priors to first-stage score (minimal diff) ---
    decision_src = ("hybrid-ml" if mode == "hybrid" else "pure-ml")

    if (mode == "hybrid") and (not getattr(args, "hybrid_priors_off", False)):
        # safe pull with defaults
        qd = sub['queue_depth'].to_numpy(np.float32, copy=False) if 'queue_depth' in sub.columns else 0.0
        qd_min, qd_max = (float(np.nanpercentile(qd, 5)) if np.size(qd) else 0.0,
                          float(np.nanpercentile(qd,95)) if np.size(qd) else 1.0)
        qdepth_norm = (qd - qd_min) / (qd_max - qd_min + 1e-9)
        qdepth_norm = np.clip(qdepth_norm, 0.0, 1.5).astype(np.float32, copy=False)
        est_transfer = sub.get('est_transfer_ms', 0.0).to_numpy(np.float32, copy=False) \
                     + sub.get('est_jitter_ms', 0.0).to_numpy(np.float32, copy=False)
        cache_hit = (sub.get('cache_hit_prob_est', 0.0).to_numpy(np.float32, copy=False) >= 0.5).astype(np.float32)
        cold_start = (sub.get('p_cold_est', 0.0).to_numpy(np.float32, copy=False) >= 0.5).astype(np.float32)

        mu_hat   = mean[sub.index].astype(np.float32, copy=False)
        tau95    = p95[sub.index].astype(np.float32, copy=False)
        sigmahat = uncert[sub.index].astype(np.float32, copy=False)
        base = (1.0 - float(getattr(args, "lambda_tail", 0.7))) * mu_hat \
             + (float(getattr(args, "lambda_tail", 0.7))) * tau95

        # priors
        score_adj = np.zeros_like(base, dtype=np.float32)
        score_adj += float(getattr(args, "edge_bias", 0.0)) * sub.get('is_edge', 0.0).to_numpy(np.float32, copy=False)
        score_adj += float(getattr(args, "uncert_penalty", 0.0)) * sigmahat
        score_adj += float(getattr(args, "qdepth_penalty", 0.0)) * qdepth_norm
        score_adj += float(getattr(args, "transfer_penalty", 0.0)) * est_transfer
        score_adj += (-float(getattr(args, "cold_penalty", 0.0)))  * cold_start
        score_adj += (-float(getattr(args, "cache_bonus", 0.0)))   * cache_hit

        # Hybrid-ML: reranker score에 priors를 더해 "동점대" 안정화
        scores = (scores + score_adj).astype(np.float32, copy=False)
        sub['score_priors'] = score_adj
        sub['priors_used'] = True
        # ← report용 임시 보관(최소침습): 최종 빌더에서 extra로 결합
        try:
            setattr(args, "_report_priors_used", True)
        except Exception:
            pass        
    else:
        sub['priors_used'] = False
        try:
            setattr(args, "_report_priors_used", False)
        except Exception:
            pass
    
    return sub, scores, "ml", decision_src
