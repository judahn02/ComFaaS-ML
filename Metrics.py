import numpy as np, pandas as pd

def oracle_selection(df_realized: pd.DataFrame):
    idx = df_realized.groupby('req_id')['total_latency_ms_realized'].idxmin()
    return df_realized.loc[idx, ['req_id','node_id','total_latency_ms_realized']].rename(
        columns={'node_id':'oracle_node','total_latency_ms_realized':'oracle_latency_ms'})

def eval_match_p95(winners: pd.DataFrame, realized: pd.DataFrame):
    merged = winners[['req_id','node_id']].merge(
        realized[['req_id','node_id','total_latency_ms_realized']],
        on=['req_id','node_id'], how='left')
    orc = oracle_selection(realized)
    merged = merged.merge(orc, on='req_id', how='left')
    match = (merged['node_id'] == merged['oracle_node']).mean()*100.0
    gap = merged['total_latency_ms_realized'] - merged['oracle_latency_ms']
    p95_gap = float(np.percentile(gap.values, 95))
    return {"oracle_match_pct": float(match), "p95_decision_gap_ms": p95_gap}

def topk_oracle_recall(pruned: pd.DataFrame, realized: pd.DataFrame):
    orc = oracle_selection(realized)
    joined = orc.merge(pruned[['req_id','node_id']], left_on=['req_id','oracle_node'],
                       right_on=['req_id','node_id'], how='left')
    return float(joined['node_id'].notna().mean()*100.0)

