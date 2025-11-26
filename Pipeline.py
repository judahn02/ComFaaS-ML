# Pipeline.py (ComFaaS_ML_50.1.3)

import os, time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from SafeSelect import SafeSelector

from Print import log, set_seed, save_df, save_json
from Data import CSVSource, SyntheticSource, attach_realized_latency, build_proxy_targets
from Features import FeatureSchema, RegressorBlock, RerankerBlock
from Regressor import MLPRegressor
from Pruning import SortPruner, PruneCfg
from Reranker import ListNetReranker
import Metrics as M

# === v80.1.2 modular skeleton: do not change behaviors! ===
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict
import pandas as pd

from pipeline.common import stage_timer
from pipeline.persist_ops import _hyb_persist_outputs

from pipeline.common import emit_cfg_once

# common.py
from pipeline.common import META_VERSION, _force_determinism, _values_for_sub, _round2, _jsonify, DEVICE

# [A][B] regressor_ops.py
from pipeline.regressor_ops import _hyb_reg_train, _hyb_reg_infer

# [C] prune_stage.py
from pipeline.prune_stage import _hyb_prune_and_realize

# [D] rerank_stage.py
from pipeline.rerank_stage import _hyb_train_reranker_if_needed

# [E][F] scoring_ops.py
from pipeline.scoring_ops import _hyb_score_heur, _hyb_score_ml

# [G] cand_builder.py
from pipeline.cand_builder import _hyb_build_cand

# [H] selection_ops.py
from pipeline.selection_ops import _hyb_select, epsilon_maxscore_select

# [I][K] reporting_ops.py
from pipeline.reporting_ops import _hyb_dual_reporting, _hyb_print_banners

# [J] eps_stats.py
from pipeline.eps_stats import _hyb_eps_stats

# [L] report_io.py
from pipeline.report_io import _hyb_build_report_dict

# [M] persist_ops.py
from pipeline.persist_ops import _hyb_persist_outputs


@dataclass
class _Result:
    pruned: pd.DataFrame
    winners: pd.DataFrame
    realized: pd.DataFrame
    report: Dict[str, Any]


torch.set_float32_matmul_precision("high") if hasattr(torch, "set_float32_matmul_precision") else None


# ==== 80.1.2 report.json schema normalization ================================
# 논문/도구 스니펫에서 공통으로 파싱 가능한 "정규 키 이름"을 고정.
# 값이 없을 경우 None(null)로 채워 일관된 필드 존재 보장.

REPORT_REQUIRED_KEYS = [
    # 메타
    "Mode", "Decision", "Seed",
    # 리콜
    "Top-K recall",
    # Heur (pre-ML)
    "Oracle match (Hybrid-Heur, pre-ML)",
    "p95 gap (Hybrid-Heur, pre-ML)",
    # Pure-Heur (선택적; 있을 때 채움)
    "Oracle match (Pure-Heur)",
    "p95 gap (Pure-Heur)",
    # Hybrid-ML (reranked)
    "Oracle match (Hybrid-ML, reranked)",
    "p95 gap (Hybrid-ML, reranked)",
    # Pure-ML (reranked)
    "Oracle match (Pure-ML, reranked)",
    "p95 gap (Pure-ML, reranked)",
    # epsilon
    "ε-trigger (full 20)",
    "ε-trigger (post-prune)",
    # 러닝 파라미터(논문 표용)
    "K", "lambda_tail", "eps",
    # 시간(벽시계 분해)
    "A) Regressor train(s)",
    "B) Regressor infer(s)",
    "C) Prune+Realized+Recall(s)",
    "D) Reranker train(s)",
    "E) Heuristic scoring(s)",
    "G) Build candidate(s)",
    "H) Final selection(s)",
    "I) Dual reporting(s)",
    "J) epsilon-trigger stats(s)",
    "K) print banners(s)",
    "L) build & save report.json(s)",
    "M) persist(s)",
    "Total(s)",
]

def _as_float_or_none(x):
    try:
        # bool 방지: True→1.0이 되지 않게 명시
        if isinstance(x, bool):
            return float(1) if x else float(0)
        return float(x)
    except Exception:
        return None

def normalize_report_schema(report: dict, *, mode, decision, recall, args, times_dict=None) -> dict:
    """
    - report: 기존 _hyb_build_report_dict(...) 가 만든 dict (부분 필드/레이블 섞여 있어도 됨)
    - mode/decision/recall/args: L 스테이지에서 이미 갖고 있는 로컬 값
    - times_dict: stage_timer가 남긴 타이밍 dict (없으면 None 허용)
    결과: REPORT_REQUIRED_KEYS 모든 키를 포함하는 dict (없으면 None)
    """

    out = dict(report or {})  # 기존 값을 최대한 보존

    # 1) 메타/공통
    out["Mode"] = out.get("Mode", mode)
    out["Decision"] = out.get("Decision", decision)
    if recall is not None:
        out["Top-K recall"] = _as_float_or_none(out.get("Top-K recall", recall))
    # seed, 하이퍼파라미터
    if hasattr(args, "seed"):
        out["Seed"] = _as_float_or_none(out.get("Seed", args.seed))
    if hasattr(args, "K"):
        out["K"] = _as_float_or_none(out.get("K", args.K))
    if hasattr(args, "lambda_tail"):
        out["lambda_tail"] = _as_float_or_none(out.get("lambda_tail", args.lambda_tail))
    if hasattr(args, "eps"):
        out["eps"] = _as_float_or_none(out.get("eps", args.eps))

    # 2) 점수 라벨 표준화 (동일한 문자열로 강제)
    # Heur (pre-ML)
    if "Oracle match (Hybrid-Heur, pre-ML)" not in out:
        # alias 흡수: 기존 코드가 "Oracle match (heur)" 등으로 썼다면 여기서 이식
        for alias in ["Oracle match (heur)", "Oracle match (Hybrid-Heur)", "oracle_hybrid_heur"]:
            if alias in out:
                out["Oracle match (Hybrid-Heur, pre-ML)"] = _as_float_or_none(out[alias])
                break
    if "p95 gap (Hybrid-Heur, pre-ML)" not in out:
        for alias in ["p95 gap (heur)", "p95 gap (Hybrid-Heur)", "p95_hybrid_heur"]:
            if alias in out:
                out["p95 gap (Hybrid-Heur, pre-ML)"] = _as_float_or_none(out[alias])
                break

    # Pure-Heur (선택)
    if "Oracle match (Pure-Heur)" not in out:
        for alias in ["oracle_pure_heur", "Oracle match (Pure Heur)"]:
            if alias in out:
                out["Oracle match (Pure-Heur)"] = _as_float_or_none(out[alias])
                break
    if "p95 gap (Pure-Heur)" not in out:
        for alias in ["p95_pure_heur", "p95 gap (Pure Heur)"]:
            if alias in out:
                out["p95 gap (Pure-Heur)"] = _as_float_or_none(out[alias])
                break

    # Hybrid-ML
    if "Oracle match (Hybrid-ML, reranked)" not in out:
        for alias in ["Oracle match (ML, reranked)", "oracle_hybrid_ml", "Oracle match (Hybrid-ML)"]:
            if alias in out:
                out["Oracle match (Hybrid-ML, reranked)"] = _as_float_or_none(out[alias])
                break
    if "p95 gap (Hybrid-ML, reranked)" not in out:
        for alias in ["p95 gap (ML, reranked)", "p95_hybrid_ml", "p95 gap (Hybrid-ML)"]:
            if alias in out:
                out["p95 gap (Hybrid-ML, reranked)"] = _as_float_or_none(out[alias])
                break

    # Pure-ML
    if "Oracle match (Pure-ML, reranked)" not in out:
        for alias in ["oracle_pure_ml", "Oracle match (Pure-ML)"]:
            if alias in out:
                out["Oracle match (Pure-ML, reranked)"] = _as_float_or_none(out[alias])
                break
    if "p95 gap (Pure-ML, reranked)" not in out:
        for alias in ["p95_pure_ml", "p95 gap (Pure-ML)"]:
            if alias in out:
                out["p95 gap (Pure-ML, reranked)"] = _as_float_or_none(out[alias])
                break

    # epsilon
    if "ε-trigger (full 20)" not in out:
        for alias in ["epsilon-trigger (full 20)", "eps_full20"]:
            if alias in out:
                out["ε-trigger (full 20)"] = _as_float_or_none(out[alias])
                break
    if "ε-trigger (post-prune)" not in out:
        for alias in ["epsilon-trigger (post-prune)", "eps_post"]:
            if alias in out:
                out["ε-trigger (post-prune)"] = _as_float_or_none(out[alias])
                break

    # 3) 타이밍(있으면 흡수; 없으면 None)
    td = times_dict or {}
    def pull_time(label, *aliases):
        if label in out and isinstance(out[label], (int, float)):
            return
        for k in (label,)+aliases:
            v = td.get(k)
            if isinstance(v, (int,float)):
                out[label] = float(v); return
        # 최종 None은 아래에서 채움

    pull_time("A) Regressor train(s)", "A")
    pull_time("B) Regressor infer(s)", "B")
    pull_time("C) Prune+Realized+Recall(s)", "C")
    pull_time("D) Reranker train(s)", "D")
    pull_time("E) Heuristic scoring(s)", "E")
    pull_time("G) Build candidate(s)", "G")
    pull_time("H) Final selection(s)", "H")
    pull_time("I) Dual reporting(s)", "I")
    pull_time("J) epsilon-trigger stats(s)", "J")
    pull_time("K) print banners(s)", "K")
    pull_time("L) build & save report.json(s)", "L")
    pull_time("M) persist(s)", "M")
    pull_time("Total(s)", "TOTAL", "Total")

    # 4) 누락 키는 전부 None으로 메울 것
    for k in REPORT_REQUIRED_KEYS:
        out.setdefault(k, None)

    return out
# ==== /report.json schema normalization =====================================

def _num_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

def _pick(d, *names):
    """여러 후보 키 이름 중 먼저 발견되는 숫자 값을 반환(없으면 None)"""
    if not isinstance(d, dict): return None
    lower = {k.lower(): v for k,v in d.items()}
    for n in names:
        v = lower.get(n.lower())
        if isinstance(v, (int, float)): 
            return float(v)
    return None

def _run_hybrid_like(args, clean: pd.DataFrame, mode_override=None, train_rr_override=None):
    # This is the original HYBRID / PURE-ML path body, preserved verbatim;
    # only 'mode' and 'train_rr' can be overridden to route to a specific submode.
    mode = getattr(args, 'mode', 'hybrid')
    if mode_override is not None:
        mode = mode_override

    # ---------- HYBRID / PURE-ML path (31.1.1 behavior preserved) ----------

    _force_determinism(int(getattr(args, "seed", 0)) + 7)

    # === [A) Regressor train — START] =======================================
    with stage_timer("A) Regressor train"):
        reg, scaler = _hyb_reg_train(args, clean)
    # === [A) Regressor train — END] =========================================

    # === [B) Regressor infer — START] =======================================
    with stage_timer("B) Regressor infer"):
        mean, p95, uncert = _hyb_reg_infer(reg, scaler, clean, args)
    # === [B) Regressor infer — END] =========================================

    # === [C) Prune + Realized + Top-K recall — START] =======================
    with stage_timer("C) Prune+Realized+Recall"):
        pruned, realized, recall, t_prune, extra = _hyb_prune_and_realize(clean, mean, p95, uncert, args)
    # === [C) Prune + Realized + Top-K recall — END] =========================

    # === [D) Train reranker (optional) — START] =============================
    with stage_timer("D) Reranker train (optional)"):
        reranker = _hyb_train_reranker_if_needed(pruned, realized, mean, p95, uncert, args, mode, train_rr_override)
    # === [D) Train reranker (optional) — END] =============================

    # === [E/F) Scoring (ML vs Heur) — START] ================================
    # 8) Rerank & Select (ε-tie + edge bias) ----------------------------------
    ts0 = time.time()
    sub = pruned.copy()

    if reranker is not None:
        # # ADDED: prevent grad/graph creation during inference for determinism
        with stage_timer("E) Heuristic scoring"):
            sub, scores, score_src, decision_src = _hyb_score_ml(reranker, sub, mean, p95, uncert, args, mode)
    else:
        # 31.1.1 fallback heuristic scoring (tail-margin 포함)
        with stage_timer("F) ML scoring"):
            sub, scores, score_src, decision_src = _hyb_score_heur(sub, mean, p95, uncert, args)

    ts1 = time.time()
    log(f"[Score] time={ts1 - ts0:.2f}s")
    # === [E/F) Scoring (ML vs Heur) — END] ==================================

    # === [G) Build candidate table — START] =================================
    with stage_timer("G) Build candidate table"):
        cand = _hyb_build_cand(sub, mean, p95, scores, score_src, decision_src, args)
    # === [G) Build candidate table — END] ===================================

    # === [H) Final selection — START] =======================================
    # ε-tie selection --------------------------------------------------------
    with stage_timer("H) Final selection"):
        winners = _hyb_select(cand, decision_src, args)
    # === [H) Final selection — END] =========================================

    # === [I) Side-by-side reporting (heur vs ml) — START] ===================
    # 31.1.1-style: side-by-side reporting -----------------------------------
    # H 결과를 보관
    winners_H = winners
    
    with stage_timer("I) Dual reporting (heur/ml)"):
        # ★ 여기서 oracle을 한 번만 만든다 (reporting_ops에서 재사용)
        oracle = M.oracle_selection(realized)

        # ML 경로일 때만 H 결과를 재사용(= 재선택 생략)
        ml_hint = winners_H if decision_src in ("hybrid-ml", "pure-ml") else None

        # 기존 호출부를 아래처럼(oracle=oracle 인자만 추가)
        heur_rep, ml_rep = _hyb_dual_reporting(cand, sub, reranker, realized, mean, 
                                               p95, uncert, args, oracle=oracle, ml_winners_hint=ml_hint)
    # === [I) Side-by-side reporting (heur vs ml) — END] =====================

    # I 구간 바로 아래, K 구간 시작 전에
    use_ml = (ml_rep is not None)
    
    # === [J) ε-trigger stats (full & post-prune) — START] ===================
    # ε-trigger rates (full vs post-prune) -----------------------------------
    with stage_timer("J) epsilon-trigger stats"):
        eps_full_pct, eps_post_pct = _hyb_eps_stats(clean, cand, mean, args)
        extra.update({
            "epsilon_trigger_rate_pct": eps_full_pct,
            "epsilon_trigger_rate_post_pct": eps_post_pct,
        })
    # === [J) ε-trigger stats (full & post-prune) — END] =====================

    # === [K) Console banners / labels — START] ==============================
    with stage_timer("K) print banners"):
        decision_lbl = _hyb_print_banners(mode, decision_src, recall, heur_rep, ml_rep, extra)
    # === [K) Console banners / labels — END] ================================

    # === [L) Build & save report.json — START] ==============================
    # with stage_timer("L) build & save report.json"):
    #     reranker_used = (ml_rep is not None)
    #     report = _hyb_build_report_dict(mode, decision_lbl, recall, heur_rep, ml_rep, reranker_used, extra, args)

    # === [FINAL] extra 합치기 & 콘솔 한 줄 ===
    # 1) 앞 단계에서 args 임시 필드로 실어둔 값 회수
    try:
        if hasattr(args, "_report_priors_used"):
            extra["priors_used"] = bool(getattr(args, "_report_priors_used"))
        if hasattr(args, "_report_tie_policy"):
            extra["tie_policy"] = getattr(args, "_report_tie_policy")
        if hasattr(args, "_report_rr_featset"):
            extra["rr_featset"] = getattr(args, "_report_rr_featset")
    except Exception:
        pass

    # 2) 디폴트 보정(없을 때 기본값)
    extra.setdefault("priors_used", False)
    extra.setdefault("tie_policy", "neutral")
    extra.setdefault("rr_featset", "ml-only")

    # 3) 콘솔 요약(한 줄)
    # --- one-time CFG summary emitter ----
    emit_cfg_once(extra)

    # # 4) report 생성 (extra를 반드시 넘김)
    # report = _hyb_build_report_dict(
    #     mode, decision_lbl, recall, heur_rep, ml_rep, reranker_used, extra, args
    # )
    # # (삭제) 첫 번째 report 호출은 reranker_used 미정의 → L 스테이지 내부에서 1회만 생성

    with stage_timer("L) build & save report.json"):
        reranker_used = (ml_rep is not None)
        report = _hyb_build_report_dict(
            mode, decision_lbl, recall, heur_rep, ml_rep, reranker_used, extra, args
        )

        # (옵션) times_dict 가져오기
        times_dict = None
        try:
            if isinstance(extra, dict):
                times_dict = extra.get("times", None)
        except Exception:
            times_dict = None

        # 80.1.2 표준 스키마로 정규화
        report = normalize_report_schema(
            report,
            mode=mode,
            decision=decision_lbl,
            recall=recall,
            args=args,
            times_dict=times_dict
        )

        # ✅ 여기서 실제 파일로 저장 (Pure-Heur 경로와 동일한 동작 보장)
        save_json(report, os.path.join(args.outdir, 'report.json'))

    # === [L) Build & save report.json — END] ================================

    # === [M) Persist CSVs + Winners head/return — START] ====================
    # --- Save CSVs (unchanged) ---
    with stage_timer("M) persist CSVs + winners head"):
        _hyb_persist_outputs(pruned, winners, realized, args, decision_lbl)
    # === [M) Persist CSVs + Winners head/return — END] ======================

    return _Result(pruned=pruned, winners=winners, realized=realized, report=report)



class PureHeurRunner:
    def execute(self, args, clean: pd.DataFrame) -> _Result:
        mode = "pure-heur"
        # NOTE: 50.1.2에도 pure-heur 코드가 있었지만, 미정의 변수 `sub` 참조 등 버그가 있어 정리함.
        from Heuristics import HeurPruner, HeurScorer

        # 3H) Heuristic pruning (no ML) --------------------------------------
        t0 = time.time()
        pruned = HeurPruner.select(
            df_clean=clean,
            K=args.K,
            params=dict(
                w_transfer=args.heur_w_transfer,
                w_jitter=args.heur_w_jitter,
                w_rho=args.heur_w_rho,
                w_mem=args.heur_w_mem
            )
        )
        t_prune = time.time() - t0

        kept = int(len(pruned))
        total = int(len(clean))
        reqs = int(clean['req_id'].nunique())
        kpr = kept / max(reqs, 1)
        log(f"[Prune] kept={kept}/{total} (~{kpr:.0f}/req) | Wall time: {t_prune:.2f} s")

        # 4H) Heuristic scoring (no ML predictions available here) -----------
        # mean_ms/p95_ms/uncert=None 로 전달 → tail_margin/uncert 패널티는 0 처리
        scores = HeurScorer.score(
            pruned_df=pruned,
            params=dict(
                lambda_tail=float(getattr(args, 'lambda_tail', 0.0)),
                uncert_penalty=float(getattr(args, 'uncert_penalty', 0.0)),
                risk_rho2=float(getattr(args, 'risk_rho2', 0.0)),
                risk_omem2=float(getattr(args, 'risk_omem2', 0.0)),
            ),
            mean_ms=None,
            p95_ms=None,
            uncert=None
        )

        # 5H) Assemble candidates --------------------------------------------
        est_total_ms = (
            pruned['est_transfer_ms'].to_numpy(float)
            + pruned['est_jitter_ms'].to_numpy(float)
            + pruned['p_cold_est'].to_numpy(float) * pruned['est_cold_ms'].to_numpy(float)
        )
        p95_total_ms = est_total_ms  # pure-heur: p95 proxy == mean proxy
        # cand = pruned[['req_id','node_id','is_edge','rho_proxy','over_mem_pos']].copy()
        cand = pruned[['req_id','node_id','is_edge']].copy()

        # --- robustify cand columns: add rho_proxy / over_mem_pos if missing ---
        if 'rho_proxy' not in cand.columns:
            if 'rho' in pruned.columns:
                cand['rho_proxy'] = pruned['rho'].to_numpy(dtype=float)
            elif 'rho_proxy' in pruned.columns:
                cand['rho_proxy'] = pruned['rho_proxy'].to_numpy(dtype=float)
            else:
                cand['rho_proxy'] = 0.0

        if 'over_mem_pos' not in cand.columns:
            _alts = [n for n in ['overmem_proxy','overmem_pos','over_mem','overmem'] if n in pruned.columns]
            cand['over_mem_pos'] = pruned[_alts[0]].to_numpy(dtype=float) if _alts else 0.0

        cand['est_total_ms'] = est_total_ms
        cand['p95_total_ms'] = p95_total_ms
        cand['score'] = scores
        cand['score_src'] = "heur"
        cand['decision_src'] = "pure-heur"

        if getattr(args, "dump_cand", False):
            save_df(cand, os.path.join(args.outdir, 'cand_pure_heur.csv'))

        # 6H) Safe select (epsilon tie + edge bias) ---------------------------
        with stage_timer("H) Final selection"):
            winners = SafeSelector.select(
                cand_df=cand,
                eps=args.eps,
                edge_bias=args.edge_bias,
                rho_thr=float(getattr(args, 'rho_thr', 0.90)),
                rho_penalty=float(getattr(args, 'rho_penalty', 2.0))
            )

        # 7H) Realized & Metrics ----------------------------------------------
        realized = attach_realized_latency(clean, seed=args.seed + 13)
        heur_report = M.eval_match_p95(winners, realized)
        recall = M.topk_oracle_recall(pruned, realized)

        extra = {"mode": mode, "rerank_used": False, "ml_compute": False, "score_src": "heur"}
        # save_df(winners, os.path.join(args.outdir, 'winners.csv'))
        # save_df(realized, os.path.join(args.outdir, 'realized.csv'))

        # --- epsilon trigger rates (pure-heur도 동일 방식으로 기록) ---
        with stage_timer("J) epsilon-trigger stats"):
            def _eps_trigger_rate(cdf: pd.DataFrame, eps: float) -> float:
                cnt = 0
                trig = 0
                for _, g in cdf.groupby('req_id', sort=False):
                    if len(g) < 2:
                        continue
                    g2 = g.sort_values('est_total_ms', ascending=True, kind='mergesort').reset_index(drop=True)
                    v = g2['est_total_ms'].to_numpy(dtype=float)
                    if (v[1] - v[0]) <= float(eps):
                        trig += 1
                    cnt += 1
                return (trig / cnt) if cnt else 0.0

            # full(원본)과 post-prune(후보)에서 est_total_ms를 일관 계산
            df_full_for_eps = clean[['req_id','est_transfer_ms','est_jitter_ms','p_cold_est','est_cold_ms']].copy()
            df_full_for_eps['est_total_ms'] = (
                df_full_for_eps['est_transfer_ms'].to_numpy(float)
                + df_full_for_eps['est_jitter_ms'].to_numpy(float)
                + df_full_for_eps['p_cold_est'].to_numpy(float) * df_full_for_eps['est_cold_ms'].to_numpy(float)
            )

            # post-prune은 '후보 집합(pruned)'에서 계산해야 req당 top-2 비교가 가능
            cand_for_eps = pruned[['req_id','est_transfer_ms','est_jitter_ms','p_cold_est','est_cold_ms']].copy()
            cand_for_eps['est_total_ms'] = (
                cand_for_eps['est_transfer_ms'].to_numpy(float)
                + cand_for_eps['est_jitter_ms'].to_numpy(float)
                + cand_for_eps['p_cold_est'].to_numpy(float) * cand_for_eps['est_cold_ms'].to_numpy(float)
            )

            eps_full = _eps_trigger_rate(df_full_for_eps[['req_id','est_total_ms']], args.eps)
            eps_post = _eps_trigger_rate(cand_for_eps[['req_id','est_total_ms']], args.eps)
            extra.update({
                "epsilon_trigger_rate_pct": float(eps_full * 100.0),
                "epsilon_trigger_rate_post_pct": float(eps_post * 100.0),
            })

        # === 80.1.2: standardized report.json ===
        decision_lbl = "Pure-Heur"
        report = {
            "topk_recall_pct": _round2(recall),   # ← 반올림 적용
            "pure_heur": {
                "oracle_match_pct": _round2(heur_report.get("oracle_match_pct", 0.0)),
                "p95_decision_gap_ms": _round2(heur_report.get("p95_decision_gap_ms", 0.0))
            },
            "meta": {
                "mode": mode,
                "reranker": False,
                "decision": decision_lbl,
                # (선택) 실행 메타 보강
                "version": META_VERSION,
                "args": dict(vars(args))  # 직렬화 가능한 기본 타입만 포함됨
            },
            "schema": "80.1.2",          # ← 스키마 버전 추가
            "extra": extra               # ← 딕셔너리 내부에 두는 편이 깔끔
        }

        # make meta.args JSON-serializable (optional but robust)
        try:
            report.setdefault("meta", {})
            report["meta"]["args"] = {k: _jsonify(v) for k, v in vars(args).items()}
        except Exception:
            pass

        save_json(report, os.path.join(args.outdir, 'report.json'))

        # --- M) persist CSVs + winners head (표준 persist; 여기선 헤드 출력 OFF) ---
        with stage_timer("M) persist CSVs + winners head"):
            _hyb_persist_outputs(
                pruned=pruned,
                winners=winners,
                realized=realized,
                args=args,
                decision_lbl="Pure-Heur",
                print_head=False  # ← M에서 헤드 로그 끔 (네가 아래에서 직접 찍음)
            )


        # normalize % to fraction if someone upstream already returned percent
        def _as_frac(x: float) -> float:
            x = float(x)
            return x/100.0 if x > 1.0 else x

        _topk = _as_frac(recall)
        _om   = _as_frac(heur_report.get('oracle_match_pct', 0.0))
        _p95  = float(heur_report.get('p95_decision_gap_ms', 0.0))

        # === 80.1.2 console labels (pure-heur) ===
        log("\n=== ComFaaS-ML 80.1.2 — Results (labels v80.1.2) ===")
        log(f"Mode: {mode} | Decision: Pure-Heur")

        # 지표
        log(f"Top-K recall: {_topk*100:.2f}%")
        log(f"Oracle match (Pure-Heur): {_om*100:.2f}% | p95 gap (Pure-Heur): {_p95:.2f} ms")

        # (옵션) extra에 ε-trigger가 있으면 full/post 모두 출력
        if 'epsilon_trigger_rate_pct' in extra and 'epsilon_trigger_rate_post_pct' in extra:
            log(f"ε-trigger (full 20): {extra['epsilon_trigger_rate_pct']:.2f}% | "
                f"(post-prune): {extra['epsilon_trigger_rate_post_pct']:.2f}%")

        # Winners 미리보기
        log("\n=== Winners (Pure-Heur) — head ===")
        log(winners.head(10).to_string(index=False))

        return _Result(pruned=pruned, winners=winners, realized=realized, report=report)

class HybridHeurRunner:
    def execute(self, args, clean: pd.DataFrame) -> _Result:
        # Force classic Hybrid-Heur path (no reranker)
        return _run_hybrid_like(args, clean, mode_override='hybrid', train_rr_override=False)

class HybridMLRunner:
    def execute(self, args, clean: pd.DataFrame) -> _Result:
        # Force Hybrid-ML path (reranker on)
        return _run_hybrid_like(args, clean, mode_override='hybrid', train_rr_override=True)

class PureMLRunner:
    def execute(self, args, clean: pd.DataFrame) -> _Result:
        # Force Pure-ML path (reranker on)
        return _run_hybrid_like(args, clean, mode_override='pure-ml', train_rr_override=True)

def _build_runner(args):
    if getattr(args, 'mode', 'hybrid') == 'pure-heur':
        return PureHeurRunner()
    if getattr(args, 'mode', 'hybrid') == 'hybrid':
        return HybridMLRunner() if getattr(args, 'train_reranker', False) else HybridHeurRunner()
    if getattr(args, 'mode', 'hybrid') == 'pure-ml':
        return PureMLRunner()
    raise ValueError(f"Unknown mode: {getattr(args, 'mode', None)}")

def run(args):
    set_seed(args.seed)
    log(f"Using device: {DEVICE}")

    _force_determinism(int(getattr(args, "seed", 0)))

    # 1) Load -----------------------------------------------------------------
    if getattr(args, "csv", None):
        raw = CSVSource(args.csv).load()
        log(f"[Load] CSV rows={len(raw)} from {args.csv}")
    else:
        raw = SyntheticSource(args.reqs, args.nodes, args.seed).load()
        log(f"[Synth] rows={len(raw)}")

    # 2) Schema sanitize (+ inverse crosses in Features) ----------------------
    clean = FeatureSchema().sanitize(raw)

    # (mode switch) -----------------------------------------------------------
    mode = getattr(args, "mode", "hybrid")

    # 80.1.2: ensure 'extra' exists in ALL paths (pure-heur / hybrid / pure-ml)
    extra = {}

    # ---------- PURE-HEUR path: skip ML entirely ----------
    # === Modular dispatch (modes) ===
    runner = _build_runner(args)
    out = runner.execute(args, clean)
    return out

def _epsilon_safe_select(*args, **kwargs):
    # Deprecated: use SafeSelector.select(...)
    raise RuntimeError("Deprecated: use SafeSelector.select(...)")