# pipeline/selection_ops.py

import os, sys, time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# 공통 유틸(필요 시)
from .common import log

# 선택 로직 제공자:
from SafeSelect import SafeSelector

# --- add: pure NumPy segment selector ---
import numpy as np
import pandas as pd


#-------- Utility Functions -------------------

def _col_or_default(df: pd.DataFrame, name: str, default, dtype, size: int):
    """
    df[name]가 있으면 ndarray로, 없으면 default(스칼라/상수)를 size 길이로 broadcast해 반환.
    dtype은 강제.
    """
    if name in df.columns:
        return df[name].to_numpy(dtype, copy=False)
    # default가 스칼라일 때 size로 확장
    arr = np.asarray(default, dtype=dtype)
    if arr.ndim == 0:
        return np.full(size, arr.item(), dtype=dtype)
    # 혹시 길이 1짜리 벡터여도 size로 확장
    if arr.size == 1 and size is not None and size != 1:
        return np.full(size, arr.item(), dtype=dtype)
    return arr.astype(dtype, copy=False)


# numpy 버전
def _epsilon_maxscore_select_numpy(cand_df: pd.DataFrame,
                                   eps: float,
                                   edge_bias: float,
                                   rho_thr: float,
                                   rho_penalty: float,
                                   mode: str = "hybrid") -> pd.DataFrame:
    """
    numpy 경로: ε-window 내에서 tie-break 정책을 적용해 req별 1개 winner 선택.
    - mode == "hybrid" → 보수 정책(큐깊이/RTT/cache/edge 우선)
    - mode == "pure-ml" → ML 친화 정책(tau95_hat → mu_hat)
    """
    if cand_df.empty:
        return cand_df.iloc[0:0][['req_id', 'node_id']]

    # 필수 기본 열
    req = cand_df['req_id'].to_numpy(np.int64, copy=False)
    nid = cand_df['node_id'].to_numpy(np.int64, copy=False)

    # 점수/est_total 기본값 구성
    # score 없으면 -est_total_ms(없으면 0) 기반
    est = _col_or_default(cand_df, 'est_total_ms', 0.0, np.float64, req.size)
    if 'score' in cand_df.columns:
        score = cand_df['score'].to_numpy(np.float64, copy=False)
    else:
        score = -est

    # (옵션) edge bias 가중
    is_edge = _col_or_default(cand_df, 'is_edge', 0.0, np.float64, req.size)
    if edge_bias:
        score = score + float(edge_bias) * is_edge

    # tie-break에 쓰일 부가 피처들(모두 안전 배열화)
    # Hybrid용
    qdepth = _col_or_default(cand_df, 'qdepth_norm', 1.0, np.float64, req.size)
    est_transfer = _col_or_default(cand_df, 'est_transfer_ms', 0.0, np.float64, req.size)
    est_jitter   = _col_or_default(cand_df, 'est_jitter_ms', 0.0, np.float64, req.size)
    cache_hit    = _col_or_default(cand_df, 'cache_hit', 0, np.int8, req.size).astype(np.int8, copy=False)
    # Pure-ML용
    mu_hat   = _col_or_default(cand_df, 'mu_hat', 0.0, np.float64, req.size)
    tau95_hat = _col_or_default(cand_df, 'tau95_hat', 0.0, np.float64, req.size)

    # 정렬: req 우선, 그 다음 est_total(오름차순)로 창 기준을 만들고 → ε-window 안에서 score로 승자 결정
    order = np.lexsort((est, req))
    req_o, nid_o = req[order], nid[order]
    est_o = est[order]
    sc_o  = score[order]

    # 보조 피처들도 동일 정렬
    qdepth_o     = qdepth[order]
    transfer_o   = est_transfer[order]
    jitter_o     = est_jitter[order]
    cache_o      = cache_hit[order]
    is_edge_o    = is_edge[order]
    mu_hat_o     = mu_hat[order]
    tau95_hat_o  = tau95_hat[order]

    # req별 segment
    cut = np.r_[True, req_o[1:] != req_o[:-1]]
    seg_st = np.flatnonzero(cut)
    seg_en = np.r_[seg_st[1:], req_o.size]

    # 각 req에서 min(est)와 ε-window 마스크
    est_min = np.minimum.reduceat(est_o, seg_st)
    seg_id = np.repeat(np.arange(seg_st.size), seg_en - seg_st)
    near = (est_o - est_min[seg_id]) <= float(eps)

    # ε-window 밖은 점수 후보에서 제외
    sc_masked = np.where(near, sc_o, -1e300)

    # ==== [추가] 타이브레이크에서 사용할 컬럼들을 안전하게 ndarray로 준비 ====
    import numpy as _np

    # # hybrid용 후보 키들 (없으면 None로 둔다)
    # qdepth = cand_df['qdepth_norm'].to_numpy(_np.float64, copy=False)       if 'qdepth_norm'       in cand_df.columns else None
    # etrans = cand_df['est_transfer_ms'].to_numpy(_np.float64, copy=False)   if 'est_transfer_ms'   in cand_df.columns else None
    # ejitt  = cand_df['est_jitter_ms'].to_numpy(_np.float64, copy=False)     if 'est_jitter_ms'     in cand_df.columns else None
    # cache  = cand_df['cache_hit'].to_numpy(_np.float64, copy=False)         if 'cache_hit'         in cand_df.columns else None
    # edge   = cand_df['is_edge'].to_numpy(_np.float64, copy=False)           if 'is_edge'           in cand_df.columns else None

    # # pure-ml용 후보 키들 (없으면 None)
    # tau95 = cand_df['tau95_hat'].to_numpy(_np.float64, copy=False)          if 'tau95_hat' in cand_df.columns else None
    # mu    = cand_df['mu_hat'].to_numpy(_np.float64, copy=False)             if 'mu_hat'    in cand_df.columns else None

    # # 점수/비용 배열도 확실히 float64 ndarray로
    # # (여기서는 기존 코드에서 만든 sc_o(점수) / est_o(비용) 중 선택해 사용)
    # # sc_o 가 "큰 값 우선", est_o 가 "작은 값 우선" 기준이라면 아래 루프에서 sc_o만 쓰면 됩니다.
    # sc_o = sc_o.astype(_np.float64, copy=False)


    # ==== [교체] 세그먼트 루프: ε-창 내 후보가 2개 이상일 때만 타이브레이크 계산 ====
    # ==== [교체] 세그먼트 루프 (…_o 배열 사용) ====
    picks = []
    for s, e in zip(seg_st, seg_en):
        if e <= s:
            continue

        mask = near[s:e]
        if not mask.any():
            picks.append(s)
            continue

        idx = np.nonzero(mask)[0]
        if idx.size == 1:
            picks.append(s + int(idx[0]))
            continue

        jabs = s + idx  # 정렬된 배열 기준 절대 인덱스

        if mode == "hybrid":
            # 보수 정책: qdepth(작게) → RTT(작게) → cache(1) → edge(1) → score(크게)
            k_qd    = qdepth_o[jabs]
            k_rtt   = transfer_o[jabs] + jitter_o[jabs]
            k_cache = cache_o[jabs].astype(np.float64, copy=False)
            k_edge  = is_edge_o[jabs].astype(np.float64, copy=False)
            k_sc    = sc_o[jabs].astype(np.float64, copy=False)

            order_sub = np.lexsort((
                -k_sc,       # score: 크게
                -k_edge,     # edge: 1 우선
                -k_cache,    # cache: 1 우선
                k_rtt,      # RTT: 작게
                k_qd,       # qdepth: 작게 (1순위)
            ))
            picks.append(int(jabs[order_sub[0]]))
        else:
            # pure-ml: tau95(작게) → mu(작게) → score(크게)
            k_t95 = tau95_hat_o[jabs]
            k_mu  = mu_hat_o[jabs]
            k_sc  = sc_o[jabs].astype(np.float64, copy=False)

            order_sub = np.lexsort((
                -k_sc,   # score: 크게
                k_mu,   # mu: 작게
                k_t95,  # tau95: 작게 (1순위)
            ))
            picks.append(int(jabs[order_sub[0]]))


    # 최종 winners DataFrame 구성(기존 로직 유지)
    winners = pd.DataFrame({
        'req_id': req_o[picks],
        'node_id': nid_o[picks]
    }).sort_values(['req_id', 'node_id']).reset_index(drop=True)

    # 리포트용 정책 태그(기존 유지)
    winners.attrs["tie_policy"] = ("conservative" if mode == "hybrid" else "neutral")
    return winners



# pandas 버전, 미세 튜닝안 (copy 최소화 & 한 번씩만)
def _epsilon_maxscore_select_pandas(cand_df: pd.DataFrame,
                                    eps: float,
                                    edge_bias: float,
                                    rho_thr: float,
                                    rho_penalty: float,
                                    mode: str = "hybrid") -> pd.DataFrame:

    if cand_df.empty:
        return cand_df.iloc[0:0][['req_id','node_id']]

    # copy 피하고 view 위주로
    eps = float(eps)
    edge_bias = float(edge_bias)
    req = cand_df['req_id']
    est = cand_df['est_total_ms'].to_numpy(float, copy=False)

    # score 준비(없으면 -est)
    if 'score' in cand_df.columns:
        score = cand_df['score'].to_numpy(float, copy=False)
    else:
        score = -est

    # edge_bias
    if edge_bias and ('is_edge' in cand_df.columns):
        score = score + float(edge_bias) * cand_df['is_edge'].to_numpy(float, copy=False)

    if eps < 0:
        idx = pd.Series(score, index=cand_df.index).groupby(req).idxmax()
        winners = cand_df.loc[idx, ['req_id','node_id']]
        return winners.sort_values(['req_id','node_id']).reset_index(drop=True)

    # req별 최소 est
    min_est = cand_df.groupby('req_id')['est_total_ms'].transform('min').to_numpy(float, copy=False)
    mask = (est <= (min_est + eps))

    # 윈도우 내부만
    sub_idx   = cand_df.index[mask]
    sub_req   = req.iloc[mask]
    sub_score = score[mask]

    idx_win = pd.Series(sub_score, index=sub_idx).groupby(sub_req).idxmax()
    winners = cand_df.loc[idx_win, ['req_id','node_id']]

    # return winners.sort_values(['req_id','node_id']).reset_index(drop=True)
    winners = winners.sort_values(['req_id','node_id']).reset_index(drop=True)
    try:
        winners.attrs["tie_policy"] = ("conservative" if mode == "hybrid" else "neutral")
    except Exception:
        pass
    return winners


def epsilon_maxscore_select(cand_df: pd.DataFrame,
                            eps: float,
                            edge_bias: float,
                            rho_thr: float,
                            rho_penalty: float,
                            mode: str = "hybrid") -> pd.DataFrame:

    # 빠른 실험: NumPy 버전으로 위임 (필요 시 원복/토글 가능)
    # return _epsilon_maxscore_select_numpy(cand_df, eps, edge_bias, rho_thr, rho_penalty)

    impl = os.getenv("H_IMPL", "numpy")  # "numpy" or "pandas"
    if impl == "numpy":
        # return _epsilon_maxscore_select_numpy(cand_df, eps, edge_bias, rho_thr, rho_penalty)
        return _epsilon_maxscore_select_numpy(cand_df, eps, edge_bias, rho_thr, rho_penalty, mode=mode)

    else:
        # return _epsilon_maxscore_select_pandas(cand_df, eps, edge_bias, rho_thr, rho_penalty)
        return _epsilon_maxscore_select_pandas(cand_df, eps, edge_bias, rho_thr, rho_penalty, mode=mode)


# === H) Final selection ======================================================
def _hyb_select(cand: pd.DataFrame,
                decision_src: str,
                args) -> pd.DataFrame:
    """[H) Final selection] decision_src가 'hybrid'면 epsilon_maxscore_select, 아니면 SafeSelector.
    in : cand, decision_src, args
    out: winners_df
    """
    # ε-tie selection ---------------------------------------------------------
    if decision_src == "hybrid":
        # 하이브리드(리랭커 OFF): 점수 우선 + ε-가드 선택기
        winners = epsilon_maxscore_select(
            cand_df=cand,
            eps=args.eps,
            edge_bias=args.edge_bias,
            rho_thr=float(getattr(args, 'rho_thr', 0.90)),
            rho_penalty=float(getattr(args, 'rho_penalty', 2.0)),
            mode=getattr(args, "mode", "hybrid")
        )
        # ← report용 정책 임시 보관
        try:
            setattr(args, "_report_tie_policy", winners.attrs.get("tie_policy", "neutral"))
        except Exception:
            pass
    # else:
    elif decision_src == "hybrid-ml":
        # 하이브리드-ML도 ε-selector 사용 (정책은 selection_ops가 mode에 따라 분기)
        winners = epsilon_maxscore_select(
            cand_df=cand,
            eps=args.eps,
            edge_bias=args.edge_bias,
            rho_thr=float(getattr(args, 'rho_thr', 0.90)),
            rho_penalty=float(getattr(args, 'rho_penalty', 2.0)),
            mode=getattr(args, "mode", "hybrid")  # "hybrid"면 conservative, "pure-ml"면 neutral
        )
        try:
            setattr(args, "_report_tie_policy", winners.attrs.get("tie_policy", "neutral"))
        except Exception:
            pass
    else:
        # ML 사용 시(또는 다른 경로): 기존 SafeSelector 유지
        winners = SafeSelector.select(
            cand_df=cand,
            eps=args.eps,
            edge_bias=args.edge_bias,
            rho_thr=float(getattr(args, 'rho_thr', 0.90)),
            rho_penalty=float(getattr(args, 'rho_penalty', 2.0))
        )

    return winners

