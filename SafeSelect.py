# SafeSelect.py (ComFaaS_ML_50.1.2)
import numpy as np
import pandas as pd

class SafeSelector:
    """
    Mode-agnostic epsilon-tie selector.
    - Detect near ties on cand.est_total_ms for top-2.
    - Apply edge bias (if provided).
    - Keep the highest 'score' otherwise.
    """

    @staticmethod
    def select(cand_df: pd.DataFrame, eps: float, edge_bias: float,
               rho_thr: float = 0.90, rho_penalty: float = 0.0, **kwargs) -> pd.DataFrame:
        rows = []
        for rid, g in cand_df.groupby('req_id', sort=False):
            g = g.sort_values('score', ascending=False, kind='mergesort').reset_index(drop=True)
            if len(g) == 1:
                rows.append((rid, int(g.loc[0, 'node_id'])))
                continue
            i1, i2 = 0, 1
            e1, e2 = float(g.loc[i1, 'est_total_ms']), float(g.loc[i2, 'est_total_ms'])
            if abs(e2 - e1) <= eps:
                s1 = float(g.loc[i1, 'score'])
                s2 = float(g.loc[i2, 'score'])
                if edge_bias != 0.0:
                    s1 += edge_bias * float(g.loc[i1, 'is_edge'])
                    s2 += edge_bias * float(g.loc[i2, 'is_edge'])
                # Optional final nudge if utilization beyond threshold (off by default)
                if rho_penalty > 0.0:
                    s1 -= rho_penalty * max(0.0, float(g.loc[i1, 'rho_proxy']) - rho_thr)
                    s2 -= rho_penalty * max(0.0, float(g.loc[i2, 'rho_proxy']) - rho_thr)
                win = i1 if s1 >= s2 else i2
                rows.append((rid, int(g.loc[win, 'node_id'])))
            else:
                rows.append((rid, int(g.loc[0, 'node_id'])))
        return pd.DataFrame(rows, columns=['req_id','node_id'])
