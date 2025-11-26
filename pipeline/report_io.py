# pipeline/report_io.py

from typing import Dict, List, Tuple, Optional, Any
import os
import pandas as pd

from .common import log, _round2, _jsonify, META_VERSION, save_json


# === L) Build & save report.json ============================================
def _hyb_build_report_dict(mode: str,
                           decision_lbl: str,
                           recall: float,
                           heur_rep: Dict,
                           ml_rep: Optional[Dict],
                           reranker_used: bool,
                           extra: Dict,
                           args) -> Dict:
    """[L) report.json] 80.1.2 스키마로 dict 생성(meta/sections/args 포함).
    in : mode, decision_lbl, recall, heur_rep, ml_rep, reranker_used, extra, args
    out: report(dict)
    """
    # --- 80.1.2 standardized report.json ---

    use_ml = bool(reranker_used)  # 바깥 use_ml 대체

    # --------- 하이브리드 메타 필드 수집(안전 기본값 + fallback) ----------
    # 우선 extra에서 꺼내고, 없다면 args 임시 속성(fallback), 최종 기본값으로 채움
    extra = dict(extra) if isinstance(extra, dict) else {}
    priors_used = bool(
        extra.get("priors_used",
                  getattr(args, "_report_priors_used", False))
    )
    tie_policy = str(
        extra.get("tie_policy",
                  getattr(args, "_report_tie_policy", "neutral"))
    )
    rr_featset = str(
        extra.get("rr_featset",
                  getattr(args, "_report_rr_featset", "ml-only"))
    )
    # extra에도 기본값 보강(setdefault) — 리포트 consumer가 extra만 볼 수도 있어서
    extra.setdefault("priors_used", priors_used)
    extra.setdefault("tie_policy", tie_policy)
    extra.setdefault("rr_featset", rr_featset)


    report = {
        "schema": "80.1.2",
        "topk_recall_pct": _round2(recall),
        "meta": {
            "mode": mode,
            "decision": decision_lbl,
            "reranker": use_ml,
            "version": META_VERSION,
            # ---- 여기 3개 필드가 추가됨 ----
            "priors_used": priors_used,
            "tie_policy": tie_policy,
            "rr_featset": rr_featset,
        },
        "extra": extra,  # ← 그대로 유지(위에서 dict 보장)
    }

    # baseline(heur_rep) 접근을 가드형으로
    report["hybrid_heur"] = {
        "oracle_match_pct": _round2(heur_rep.get("oracle_match_pct")),
        "p95_decision_gap_ms": _round2(heur_rep.get("p95_decision_gap_ms")),
    }

    # ML 섹션(있을 때만)
    if use_ml and (ml_rep is not None):
        if mode == "hybrid":
            report["hybrid_ml"] = {
                "oracle_match_pct": _round2(ml_rep.get("oracle_match_pct")),
                "p95_decision_gap_ms": _round2(ml_rep.get("p95_decision_gap_ms")),
            }
        elif mode == "pure-ml":
            report["pure_ml"] = {
                "oracle_match_pct": _round2(ml_rep.get("oracle_match_pct")),
                "p95_decision_gap_ms": _round2(ml_rep.get("p95_decision_gap_ms")),
            }

    try:
        report["meta"]["args"] = {k: _jsonify(v) for k, v in vars(args).items()}
    except Exception:
        pass

    save_json(report, os.path.join(args.outdir, 'report.json'))

    return report
