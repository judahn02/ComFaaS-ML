# Pruning.py (ComFaaS_ML_29.1)
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class PruneCfg:
    """
    프루닝 설정 (게이트 A/B/C 합집합 → 요청별 Kmax 캡).
    - Kmax         : 요청당 최종 보존 후보 수(Top-K 상한)
    - lambda_tail  : (레거시) tail 가중, 현재는 외부에서 사용 가능
    - risk_rho     : (레거시) rho 리스크 패널티 계수 (외부)
    - risk_omem    : (레거시) over_mem 패널티 계수 (외부)
    - alpha_u      : 게이트 A에서 'uncert' 완충 계수 (↑ = 보수적)
    - beta_tail    : 게이트 A에서 tail_margin 완충 계수
    - buffer_ms    : 게이트 A에서 best LB 대비 허용 버퍼(ms)
    - m_guard      : 게이트 B에서 raw est_total 상위 보존 개수
    - c_guard      : 게이트 C에서 p95_total 상위 보존 개수
    """
    Kmax: int
    lambda_tail: float
    risk_rho: float
    risk_omem: float
    alpha_u: float = 8.0
    beta_tail: float = 0.5
    buffer_ms: float = 50.0
    m_guard: int = 10
    c_guard: int = 10


class SortPruner:
    """
    게이트 A/B/C 합집합으로 후보를 넓게 보존한 뒤,
    요청별로 Kmax로 '부분 선택(argpartition)' 하여 빠르게 캡한다.

    Gate A (LB-buffer):
      LB = est_total - alpha_u * uncert - beta_tail * tail_margin
      → req 내 best LB 대비 buffer_ms 이하 전부 keep

    Gate B (raw est_total top-m):
      → req 내 est_total 오름차순 상위 m_guard 보존

    Gate C (p95_total top-c):
      p95_total = est_transfer + est_jitter + p95 + p_cold * est_cold_ms
      → req 내 p95_total 오름차순 상위 c_guard 보존

    최종 keep = A ∪ B ∪ C  (합집합)
    if len(keep) > Kmax:
        LB 기준으로 가장 작은 Kmax개만 argpartition으로 부분 선택
    """

    def select(self, df: pd.DataFrame,
               mean: np.ndarray, p95: np.ndarray, uncert: np.ndarray,
               cfg: PruneCfg) -> pd.DataFrame:

        # --- 입력을 numpy로 (성능/안전) ---
        mean   = np.asarray(mean,   dtype=np.float32)
        p95    = np.asarray(p95,    dtype=np.float32)
        uncert = np.asarray(uncert, dtype=np.float32)

        est_transfer = df['est_transfer_ms'].to_numpy(np.float32, copy=False)
        est_jitter   = df['est_jitter_ms'].to_numpy(np.float32, copy=False)
        p_cold       = df['p_cold_est'].to_numpy(np.float32, copy=False)
        est_cold_ms  = df['est_cold_ms'].to_numpy(np.float32, copy=False)
        req          = df['req_id'].to_numpy(np.int64, copy=False)

        # --- 핵심 스코어 ---
        est_total = est_transfer + est_jitter + mean + p_cold * est_cold_ms
        tail_margin = np.maximum(0.0, p95 - mean)

        # Gate A의 완충 하한(LB): 값이 작을수록 좋음
        LB = est_total - cfg.alpha_u * uncert - cfg.beta_tail * tail_margin
        # 안전 처리(혹시 모를 NaN/Inf)
        LB = np.nan_to_num(LB, nan=np.inf, posinf=np.inf, neginf=-np.inf)

        # Gate C용 p95-based total
        p95_total = est_transfer + est_jitter + p95 + p_cold * est_cold_ms

        # --- DataFrame에 보조 컬럼 붙여 정렬키 준비 ---
        sub = df.copy()
        sub['__LB']   = LB
        sub['__ET']   = est_total
        sub['__P95T'] = p95_total

        # req asc + 각 키 asc (오름차순이 '좋음')
        order_LB   = np.lexsort((sub['__LB'  ].to_numpy(), req))
        order_ET   = np.lexsort((sub['__ET'  ].to_numpy(), req))
        order_P95T = np.lexsort((sub['__P95T'].to_numpy(), req))

        # --- 요청 경계 계산 헬퍼 ---
        def seg_bounds(req_sorted: np.ndarray):
            new_seg = np.empty_like(req_sorted, dtype=bool)
            new_seg[0] = True
            new_seg[1:] = (req_sorted[1:] != req_sorted[:-1])
            starts = np.flatnonzero(new_seg)
            ends = np.r_[starts[1:], len(req_sorted)]
            return starts, ends

        sA, eA = seg_bounds(req[order_LB])
        sB, eB = seg_bounds(req[order_ET])
        sC, eC = seg_bounds(req[order_P95T])

        keep_chunks = []
        # --- 요청별로 게이트 A/B/C 합집합을 구한 뒤, Kmax로 캡 ---
        for (sa, ea), (sb, eb), (sc, ec) in zip(zip(sA, eA), zip(sB, eB), zip(sC, eC)):
            # 세그먼트 인덱스
            segA = order_LB[sa:ea]
            segB = order_ET[sb:eb]
            segC = order_P95T[sc:ec]

            # Gate A: best LB 대비 buffer_ms 이하 전부 keep
            LB_seg = sub['__LB'].to_numpy()[segA]
            best_LB = LB_seg.min()
            keepA = segA[(LB_seg - best_LB) <= float(cfg.buffer_ms)]

            # Gate B: est_total 오름차순 상위 m_guard
            m = min(cfg.m_guard, len(segB))
            keepB = segB[:m]

            # Gate C: p95_total 오름차순 상위 c_guard
            c = min(cfg.c_guard, len(segC))
            keepC = segC[:c]

            # 합집합
            kept = np.unique(np.concatenate([keepA, keepB, keepC], axis=0))

            # 요청당 Kmax 캡 (전량 sort → argpartition으로 교체)
            if len(kept) > cfg.Kmax:
                lb_local = sub['__LB'].to_numpy()[kept]
                k = cfg.Kmax
                # 가장 작은 K개 인덱스만 부분 선택
                topk_idx = np.argpartition(lb_local, k-1)[:k]
                kept = kept[topk_idx]
                # (선택) 안정 정렬이 필요하면 아래 주석 해제
                # kept = kept[np.argsort(lb_local[topk_idx], kind='mergesort')]

            # 최악 보호: 비어 있으면 한 개는 살린다
            if kept.size == 0:
                kept = segA[:1]

            keep_chunks.append(kept)

        keep_idx = np.concatenate(keep_chunks, axis=0)

        # --- 최종 산출 ---
        kept_df = (sub.iloc[keep_idx]
                     .drop(columns=['__LB', '__ET', '__P95T'])
                     .sort_values(['req_id', 'node_id'])
                     .reset_index(drop=True))
        return kept_df
