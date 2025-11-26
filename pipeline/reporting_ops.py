# pipeline/reporting_ops.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

# 공통 유틸(heur 점수 재계산 시 _values_for_sub 사용)
from .common import log, _values_for_sub

# 메트릭은 프로젝트 관례대로 모듈 별칭 import
import Metrics as M

# 리포팅에서 휴리스틱/ML 승자 계산 시 사용
from .selection_ops import epsilon_maxscore_select

from SafeSelect import SafeSelector
from Reranker import ListNetReranker

from .common import stage_timer

# === I) Side-by-side reporting ==============================================
def _hyb_dual_reporting(cand: pd.DataFrame,
                        sub: pd.DataFrame,
                        reranker: Optional["ListNetReranker"],
                        realized: pd.DataFrame,
                        mean: np.ndarray,
                        p95: np.ndarray,
                        uncert: np.ndarray,
                        args,
                        oracle: Optional[pd.DataFrame] = None,
                        ml_winners_hint: Optional[pd.DataFrame] = None
                        ) -> Tuple[Dict, Optional[Dict]]:
    """
    [I) Side-by-side reporting]
    - E 구간에서 계산된 cand['score']를 재사용(재계산 없음)
    - Heur 후보 / ML 후보 각각 선택 후, oracle 캐시로 빠르게 메트릭 계산
    in : cand, sub, reranker, realized, mean, p95, uncert, args, (optional) oracle
    out: (heur_rep_dict, ml_rep_dict_or_None)
    """

    # ① Heur winners
    with stage_timer("I.heur_select", sync_cuda=False):
        # 점수는 E 구간에서 cand['score']로 이미 계산됨 → 여기서는 **재계산하지 않고 재사용**
        # 이 함수가 cand를 변경하지 않는 한, 얕은 참조만 써도 안전하고 메모리/시간 절약됨.
        heur_cand = cand # ← copy() 없이

        # Heuristic side (항상 epsilon)
        heur_winners = epsilon_maxscore_select(
            cand_df=heur_cand,
            eps=args.eps,
            edge_bias=args.edge_bias,
            rho_thr=float(getattr(args, 'rho_thr', 0.90)),
            rho_penalty=float(getattr(args, 'rho_penalty', 2.0))
        )

    # ② Heur eval
    with stage_timer("I.heur_eval", sync_cuda=False):
        # oracle 캐시가 있으면 재사용, 없으면 내부에서 한 번 계산하여 사용
        heur_rep = _eval_match_p95_cached(heur_winners, realized, oracle)

    # ML side (reranker 있을 때만)
    if reranker is not None:
        if ml_winners_hint is not None:
            ml_winners = ml_winners_hint
        else:
            # 이 함수가 cand를 변경하지 않는 한, 얕은 참조만 써도 안전하고 메모리/시간 절약됨.
            ml_cand = cand
            ml_winners = SafeSelector.select(
                cand_df=ml_cand,
                eps=args.eps,
                edge_bias=args.edge_bias,
                rho_thr=float(getattr(args, 'rho_thr', 0.90)),
                rho_penalty=float(getattr(args, 'rho_penalty', 2.0))
            )
        # ④ ML eval
        with stage_timer("I.ml_eval", sync_cuda=False):
            ml_rep = _eval_match_p95_cached(ml_winners, realized, oracle)
    else:
        ml_rep = None

    return heur_rep, ml_rep

# === K) Console banners / labels ============================================
def _hyb_print_banners(mode: str,
                       decision_src: str,
                       recall: float,
                       heur_rep: Dict,
                       ml_rep: Optional[Dict],
                       extra: Dict) -> str:
    """[K) Console banners] 결과 배너 출력 + 최종 decision 라벨 반환(Hybrid-Heur/Hybrid-ML/Pure-ML).
    in : mode, decision_src, recall, heur_rep, ml_rep, extra
    out: decision_label(str)
    """
    # === 80.1.2 unified console labels (hybrid / pure-ml) ===
    # decision_src: "hybrid" or "hybrid-ml" or "pure-ml" (기존 코드에서 설정됨)
    use_ml = (ml_rep is not None)
    # decision_lbl = ("Hybrid-ML" if (mode == "hybrid" and use_ml)
    #                 else ("Pure-ML" if mode == "pure-ml" else "Hybrid-Heur"))
    # === 80.1.2 unified console labels (hybrid / pure-ml) ===
    # decision_src: "hybrid" or "hybrid-ml" or "pure-ml" (기존 코드에서 설정됨)
    use_ml = (ml_rep is not None)  # 출력(N/A 가드 용도는 계속 사용
    if mode == "hybrid":
        decision_lbl = "Hybrid-Heur" if decision_src == "hybrid" else "Hybrid-ML"
    elif mode == "pure-ml":
        decision_lbl = "Pure-ML"
    else:
        # 안전장치: 예외 모드일 경우 decision_src 문자열로 보정
        decision_lbl = ("Hybrid-Heur" if decision_src == "hybrid"
                        else ("Hybrid-ML" if "ml" in decision_src else decision_src))

    # 결과 섹션 배너를 먼저 출력
    log("\n=== ComFaaS-ML 80.1.2 — Results (labels v80.1.2) ===")
    log(f"Mode: {mode} | Decision: {decision_lbl}")

    # Top-K (항상 % 포맷 0–100, 소수점 2자리)
    log(f"Top-K recall: {recall:.2f}%")

    # Baseline & final metrics: label depends on mode
    if mode == "pure-ml":
        # Baseline = Hybrid-Heur (pre-ML)
        log(f"Baseline (Hybrid-Heur, pre-ML): {heur_rep['oracle_match_pct']:.2f}% | "
            f"p95 gap (Hybrid-Heur, pre-ML): {heur_rep['p95_decision_gap_ms']:.2f} ms")
        # Final = Pure-ML (reranked)
        if use_ml:
            log(f"Oracle match (Pure-ML, reranked): {ml_rep['oracle_match_pct']:.2f}% | "
                f"p95 gap (Pure-ML, reranked): {ml_rep['p95_decision_gap_ms']:.2f} ms")
        else:
            log("Oracle match (Pure-ML, reranked): N/A | p95 gap (Pure-ML, reranked): N/A")
    else:
        # Hybrid path labels (unchanged)
        log(f"Oracle match (Hybrid-Heur, pre-ML): {heur_rep['oracle_match_pct']:.2f}% | "
            f"p95 gap (Hybrid-Heur, pre-ML): {heur_rep['p95_decision_gap_ms']:.2f} ms")

        if use_ml:
            log(f"Oracle match (Hybrid-ML, reranked): {ml_rep['oracle_match_pct']:.2f}% | "
                f"p95 gap (Hybrid-ML, reranked): {ml_rep['p95_decision_gap_ms']:.2f} ms")
        else:
            log("Oracle match (Hybrid-ML, reranked): N/A | p95 gap (Hybrid-ML, reranked): N/A")

    # ε-trigger: 계산된 경우에만 출력 (이미 extra.update로 넣었거나 지역변수 보유 시)
    if ('epsilon_trigger_rate_pct' in extra and 'epsilon_trigger_rate_post_pct' in extra):
        log(f"ε-trigger (full 20): {extra['epsilon_trigger_rate_pct']:.2f}% | "
            f"(post-prune): {extra['epsilon_trigger_rate_post_pct']:.2f}%")
    # elif ('eps_full' in locals() and 'eps_post' in locals()):
    #     # fallback: extra에 안 넣었으나 지역변수로는 있는 경우
    #     log(f"ε-trigger (full 20): {eps_full*100:.2f}% | (post-prune): {eps_post*100:.2f}%")
    # else: not computed → do not print

    return decision_lbl

# === Oracle Cache Util and Helpers ==============================================

def _build_oracle_cache(realized: pd.DataFrame) -> pd.DataFrame:
    """realized로부터 oracle_selection 한 번만 계산해 캐시를 만든다."""
    return M.oracle_selection(realized)

def _eval_match_p95_cached(winners: pd.DataFrame,
                           realized: pd.DataFrame,
                           oracle: Optional[pd.DataFrame]) -> Dict:
    """
    Metrics.eval_match_p95와 동일 결과를, 미리 계산된 oracle로 빠르게 구한다.
    oracle이 주어지면 재계산 없이 활용, 없으면 즉시 M.oracle_selection(realized)로 생성.
    반환: {"oracle_match_pct": float, "p95_decision_gap_ms": float}
    """

    # (선택) 컬럼 존재 가드 — 필요 없다면 제거 가능
    # 필수 컬럼 없으면 KeyError가 나니까, 디버깅 친화적으로 체크
    # for df, cols, name in [
    #     (winners,  ['req_id','node_id'],                                    "winners"),
    #     (realized, ['req_id','node_id','total_latency_ms_realized'],        "realized"),
    # ]:
    #     missing = [c for c in cols if c not in df.columns]
    #     if missing:
    #         raise ValueError(f"[eval_cached] {name} missing columns: {missing}")

    # # winners + realized join
    # merged = winners[['req_id','node_id']].merge(
    #     realized[['req_id','node_id','total_latency_ms_realized']],
    #     on=['req_id','node_id'], how='left'
    # )
    # winners + realized (필요 컬럼만)
    w = winners[['req_id','node_id']]
    r = realized[['req_id','node_id','total_latency_ms_realized']]
    merged = w.merge(r, on=['req_id','node_id'], how='left')

    # oracle join (캐시가 있으면 재사용, 없으면 즉시 계산)
    if oracle is None:
        oracle = M.oracle_selection(realized)
    # (선택) oracle 컬럼 가드 — 필요 없다면 제거 가능
    # for c in ['req_id','oracle_node','oracle_latency_ms']:
    #     if c not in oracle.columns:
    #         raise ValueError(f"[eval_cached] oracle missing column: {c}")

    merged = merged.merge(oracle, on='req_id', how='left')

    # 동일 정의: 매칭율(%) + 결정 p95 갭(ms)
    match = (merged['node_id'] == merged['oracle_node']).mean() * 100.0

    gap = merged['total_latency_ms_realized'] - merged['oracle_latency_ms']
    # (선택) NaN 방어 — 필요시 활성화
    # gap = gap.dropna()

    p95_gap = float(np.percentile(gap.to_numpy(), 95))

    return {"oracle_match_pct": float(match), "p95_decision_gap_ms": p95_gap}

