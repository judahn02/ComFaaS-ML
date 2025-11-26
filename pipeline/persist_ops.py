# pipeline/persist_ops.py

import os

from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import pandas as pd
import numpy as np      # 필요 없지만 일관 위해 OK (없어도 됨)

from .common import save_df, log


# === M) Persist CSVs + Winners head/return ==================================
def _hyb_persist_outputs(pruned: pd.DataFrame,
                         winners: pd.DataFrame,
                         realized: pd.DataFrame,
                         args,
                         decision_lbl: str,
                         print_head: bool = False) -> None:
    """[M) Persist] pruned/winners/realized CSV 저장(+토글) + winners head 로그."""

    # --- 저장 토글(새 플래그) ---
    save_pruned   = bool(getattr(args, "save_pruned_csv", 1))     # 기본: 저장
    save_realized = bool(getattr(args, "save_realized_csv", 1))   # 기본: 저장
    save_parquet  = bool(getattr(args, "save_parquet", False))    # 기본: OFF

    # --- CSV 저장 ---
    # winners.csv는 항상 저장(분석에 필수)
    save_df(winners,  os.path.join(args.outdir, 'winners.csv'))   # 항상
    if save_pruned:
        save_df(pruned,   os.path.join(args.outdir, 'pruned.csv'))
    if save_realized:
        save_df(realized, os.path.join(args.outdir, 'realized.csv'))

    # --- Winners head & enrich (CSV는 최소 스키마 유지, Parquet/로그는 enrich) ---
    show_cols = [
        'req_id','node_id','est_transfer_ms','est_jitter_ms','p_cold_est','est_cold_ms',
        'queue_depth','rho_proxy','over_mem_pos'
    ]
    exist_in_pruned = [c for c in show_cols[2:] if c in pruned.columns]

    # ★ winners 원본은 건드리지 않음
    w_enriched = winners.copy()
    if exist_in_pruned:
        key_pruned = list(zip(pruned['req_id'].to_numpy(), pruned['node_id'].to_numpy()))
        for c in exist_in_pruned:
            mp = dict(zip(key_pruned, pruned[c].to_numpy()))
            w_enriched[c] = list(map(
                lambda k: mp.get(k),
                zip(w_enriched['req_id'].to_numpy(), w_enriched['node_id'].to_numpy())
            ))

    safe_cols = ['req_id','node_id'] + exist_in_pruned
    head = w_enriched[safe_cols].head(10)

    # CSV/Parquet 저장(기존 로직 유지; Parquet 켜져 있으면 w_enriched 저장)
    # --- Parquet (옵션) ---
    if save_parquet:
        pruned.to_parquet(os.path.join(args.outdir, 'pruned.parquet'),   index=False)
        realized.to_parquet(os.path.join(args.outdir, 'realized.parquet'), index=False)
        # ★ enrich된 사본 저장
        w_enriched.to_parquet(os.path.join(args.outdir, 'winners.parquet'), index=False)

    if bool(getattr(args, "print_head", False)):
        log(f"\n=== Winners ({decision_lbl}) — head ===")
        log(head.to_string(index=False))
