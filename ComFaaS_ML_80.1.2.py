#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import sys
import time

# Local modules
from Print import log
import Pipeline as P


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ComFaaS-ML 80.1.2 (31.1.1 best-run defaults)",
        description="ComFaaS-ML 80.1.2 — 50.1.5 layout with 31.1.1-like defaults",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # IO / data
    p.add_argument('--csv', type=str, default=None, help='Path to CSV; if not set, use synthetic generator')
    p.add_argument('--outdir', type=str, default='80.1.2_eps015', help='Output directory')
    p.add_argument('--print_labels_style', type=str, default='80.1.2',
                  help="label style switcher (reserved for future; current=80.1.2)")
    p.add_argument('--print_head', action='store_true', default=False,
                help='Print winners head inside persist stage')

    p.add_argument('--save_pruned_csv', type=int, default=1,
                help='Save pruned.csv (1) or skip (0)')
    p.add_argument('--save_realized_csv', type=int, default=1,
                help='Save realized.csv (1) or skip (0)')

    p.add_argument('--seed', type=int, default=1337)

    # synthetic workload
    p.add_argument('--reqs', type=int, dest='reqs', default=20000)
    p.add_argument('--nodes', type=int, dest='nodes', default=20)

    # Regressor (mean & p95 heads)
    p.add_argument('--reg_epochs', type=int, default=150)
    p.add_argument('--batch', type=int, default=8192)
    p.add_argument('--reg_lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--quantile_weight', type=float, default=0.9)  # best-run override ready (31.1.1 runs used 0.9)
    p.add_argument('--reg_hidden', type=int, default=128)
    p.add_argument('--reg_dropout', type=float, default=0.1)
    p.add_argument('--mc_samples', type=int, default=20)

    # Pruning (SortPruner)
    p.add_argument('--K', type=int, default=16)
    p.add_argument('--lambda_tail', type=float, default=0.7)
    p.add_argument('--m_guard', type=int, default=10)
    p.add_argument('--c_guard', type=int, default=10)
    p.add_argument('--alpha_uncert', type=float, default=8.0)
    p.add_argument('--beta_tail', type=float, default=0.5)
    p.add_argument('--buffer_ms', type=float, default=50.0)
    p.add_argument('--risk_rho', type=float, default=6.0)
    p.add_argument('--risk_omem', type=float, default=15.0)

    # Reranker
    p.add_argument('--train_reranker', action='store_true', default=False, help='Enable reranker training (hybrid). pure-ml forces True')
    p.add_argument('--rerank_epochs', type=int, default=15)
    p.add_argument('--rerank_batch',  type=int, default=16384)
    p.add_argument('--rerank_hidden', type=int, default=128)
    p.add_argument('--rerank_dropout', type=float, default=0.1)
    p.add_argument('--rerank_lr', type=float, default=2.2e-3)
    p.add_argument('--listwise_w', type=float, default=1.0)
    p.add_argument('--pairwise_w', type=float, default=0.0)

    # 속도 모드 스위치(개발/회귀용) — 기본 OFF(정확도 영향 없음)
    p.add_argument('--fast_reranker', action='store_true', default=False,
                help='Speed mode: fewer epochs, larger batch, pairwise disabled.')

    # 플래그 추가
    p.add_argument('--save_parquet', action='store_true',
                help='Also save pruned/realized/winners as Parquet alongside CSV.')

    # cold 가중치까지 쓰려면
    p.add_argument("--heur_w_transfer", type=float, default=1.0,
                    help="Weight for est_transfer_ms in heuristic est_total")
    p.add_argument("--heur_w_jitter", type=float, default=1.0,
                    help="Weight for est_jitter_ms in heuristic est_total")
    p.add_argument("--heur_w_cold", type=float, default=1.0,
                    help="Weight for cold penalty term p_cold_est*est_cold_ms in heuristic est_total")
    p.add_argument('--heur_w_rho', type=float, default=0.0)
    p.add_argument('--heur_w_mem', type=float, default=0.0)

    # SafeSelect
    p.add_argument('--eps', type=float, default=0.15)

    # Extra penalties used in hybrid heuristic score
    p.add_argument('--risk_rho2', type=float, default=0.0)
    p.add_argument('--risk_omem2', type=float, default=0.0)
    p.add_argument('--qd_delta_ms', type=float, default=1.0,
                help='Near-tie window (ms). Queue-depth penalty applies strongly only within this est_total margin.')

    # Mode
    p.add_argument('--mode', type=str, default='hybrid', choices=['hybrid', 'pure-ml', 'pure-heur'])

    # Hybrid-ML priors (ignored in pure-ml path)
    p.add_argument('--edge_bias', type=float, default=0.10)
    p.add_argument('--uncert_penalty', type=float, default=0.30)
    p.add_argument('--qdepth_penalty', type=float, default=0.05)
    p.add_argument('--transfer_penalty', type=float, default=0.02)
    p.add_argument('--cold_penalty', type=float, default=0.50)
    p.add_argument('--cache_bonus', type=float, default=0.20)
    p.add_argument('--reranker_use_opsigs', action='store_true')

    # Ablations (hybrid only)
    p.add_argument('--hybrid_priors_off', action='store_true')
    p.add_argument('--reranker_opsigs_off', action='store_true')

    # rho guard for selection reporting
    p.add_argument('--rho_penalty', type=float, default=2.0)
    p.add_argument('--rho_thr', type=float, default=0.90)

    # debug
    p.add_argument('--dump_cand', action='store_true', default=False)
    p.add_argument('--debug', action='store_true', help='print extra debug logs')

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # 80.1.2 banner (Decision will be finalized by Pipeline)
    log("="*78)
    log("ComFaaS-ML 80.1.2 — unified labels, standardized report.json")
    log(f"Mode: {args.mode} | Outdir: {args.outdir}")
    log("="*78)
    log("Decision: (set by Pipeline after selection)")

    # Outdir prep
    os.makedirs(args.outdir, exist_ok=True)

    # Delegate to Pipeline
    t0 = time.time()
    try:
        out = P.run(args)
    except Exception as e:
        log(f"[FATAL] {type(e).__name__}: {e}")
        raise
    finally:
        log(f"[DONE] Total Wall time: {time.time() - t0:.2f} s")

    # Make it clear where outputs went
    log(f"[OUT] Saved under: {os.path.abspath(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
