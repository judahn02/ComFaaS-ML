# Features.py (ComFaaS_ML_29.1)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Feature schema & sanitization
# ---------------------------------------------------------------------
class FeatureSchema:
    """
    - CSV/합성 원본을 공통 스키마로 정리(sanitize)하고,
      여기서 inverse-structure crosses를 생성한다.
    - 생성되는 파생 컬럼:
        * hw_combo     = hardware_cpu + 0.25 * hardware_gpu
        * inv_hw_combo = 1 / (1e-6 + hw_combo)
        * io_ratio     = hardware_storage / (1e-6 + hw_combo)
        * hardware_score = hw_combo  (v26/27 동형성 유지 별칭)
    """

    # 원본 컬럼(없으면 기본값으로 백필)
    DECLARED = [
        # ids
        'req_id', 'node_id',
        # inputs / environment
        'input_size_kb',
        'est_transfer_ms', 'est_jitter_ms',
        'p_cold_est', 'est_cold_ms',
        'queue_depth', 'rho_proxy',
        'cpu_util_ema', 'mem_util_ema', 'cache_hit_prob_est',
        'is_edge', 'is_gpu',
        # hardware (3-axis; CSV에 없으면 Data.py에서 백필됨)
        'hardware_cpu', 'hardware_gpu', 'hardware_storage',
        # optional risk-ish cols (없으면 0.0으로 백필)
        'over_mem_pos',
    ]

    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = df.copy()

        # 누락 컬럼 백필
        defaults_float = {
            'input_size_kb': 0.0,
            'est_transfer_ms': 0.0, 'est_jitter_ms': 0.0,
            'p_cold_est': 0.0, 'est_cold_ms': 0.0,
            'queue_depth': 0.0, 'rho_proxy': 0.0,
            'cpu_util_ema': 0.0, 'mem_util_ema': 0.0, 'cache_hit_prob_est': 0.0,
            'hardware_cpu': 1.0, 'hardware_gpu': 1.0, 'hardware_storage': 1.0,
            'over_mem_pos': 0.0,
        }
        defaults_int = {'is_edge': 0, 'is_gpu': 0}
        for c, v in defaults_float.items():
            if c not in clean.columns:
                clean[c] = v
        for c, v in defaults_int.items():
            if c not in clean.columns:
                clean[c] = v
        if 'req_id' not in clean.columns:
            clean['req_id'] = np.arange(len(clean), dtype=np.int64)
        if 'node_id' not in clean.columns:
            clean['node_id'] = np.arange(len(clean), dtype=np.int64)

        # dtype 정리 (downstream에서 torch.float32로 캐스팅하기 좋게)
        float_cols = [
            'input_size_kb', 'est_transfer_ms', 'est_jitter_ms',
            'p_cold_est', 'est_cold_ms',
            'queue_depth', 'rho_proxy',
            'cpu_util_ema', 'mem_util_ema', 'cache_hit_prob_est',
            'hardware_cpu', 'hardware_gpu', 'hardware_storage',
            'over_mem_pos',
        ]
        for c in float_cols:
            clean[c] = clean[c].astype('float32', copy=False)
        clean['is_edge'] = clean['is_edge'].astype('int8', copy=False)
        clean['is_gpu']  = clean['is_gpu'].astype('int8', copy=False)

        # ---------- inverse-structure crosses ----------
        cpu = clean['hardware_cpu'].to_numpy('float32')
        gpu = clean['hardware_gpu'].to_numpy('float32')
        sto = clean['hardware_storage'].to_numpy('float32')

        hw_combo = cpu + 0.25 * gpu
        inv_hw_combo = 1.0 / (1e-6 + hw_combo)
        io_ratio = sto / (1e-6 + hw_combo)

        clean['hw_combo']     = hw_combo.astype('float32')
        clean['inv_hw_combo'] = inv_hw_combo.astype('float32')
        clean['io_ratio']     = io_ratio.astype('float32')
        # v26/27 동형성 유지: hardware_score = hw_combo
        clean['hardware_score'] = clean['hw_combo'].astype('float32')

        return clean


# ---------------------------------------------------------------------
# Regressor features
# ---------------------------------------------------------------------
class RegressorBlock:
    """
    회귀기(MLPRegressor) 입력 피처 구성.
    - v27.3 동형성 유지: inverse crosses 반영
      * 'hardware_score' (= hw_combo), 'inv_hw_combo', 'io_ratio'
    - 기타 운영/부하/캐시/엣지 정보 포함
    """
    COLS = [
        # workload / env
        'input_size_kb', 'queue_depth', 'rho_proxy',
        'cpu_util_ema', 'mem_util_ema', 'cache_hit_prob_est', 'is_edge',
        # inverse-structure crosses
        'hardware_score', 'inv_hw_combo', 'io_ratio',
        # (선택) 원천 3축까지 투입하고 싶으면 아래 주석 해제:
        # 'hardware_cpu','hardware_gpu','hardware_storage',
    ]

    def build(self, clean: pd.DataFrame) -> np.ndarray:
        # missing cols guard
        for c in self.COLS:
            if c not in clean.columns:
                raise KeyError(f"[RegressorBlock] missing column: {c}")
        X = clean[self.COLS].to_numpy(np.float32, copy=False)
        return X


# ---------------------------------------------------------------------
# Reranker features
# ---------------------------------------------------------------------
class RerankerBlock:
    """
    리랭커 입력 피처 구성.
    - v27.3 동형성 유지(정확히는 27.3+에서 쓰던 경량 세트):
        [-est_total, mean, tail, uncert, is_edge, rho_proxy, over_mem_pos,
         est_transfer_ms, est_jitter_ms, delta_to_min]
    - delta_to_min: 같은 req 내에서 해당 후보의 est_total - req 최소 est_total
      (top-1 근처 구분력 상승)
    - inverse crosses는 회귀 예측(mean/p95)에 이미 녹아 있으므로 여기선 중복 투입하지 않음.
    """
    FEAT_ORDER = [
        'neg_est_total',      # -(transfer + jitter + mean + p_cold*est_cold)
        'pred_mean',          # mean
        'pred_tail',          # max(0, p95 - mean)
        'pred_uncert',        # MC-dropout std
        'is_edge', 'rho_proxy', 'over_mem_pos',
        'est_transfer_ms', 'est_jitter_ms',
        'delta_to_min',       # est_total - group_min(est_total)
    ]

    def build(self,
            dfK: pd.DataFrame,
            mean: np.ndarray,
            p95: np.ndarray,
            uncert: np.ndarray,
            mode: str = "hybrid",
            use_opsigs: bool = False) -> np.ndarray:
        """
        dfK: 프루닝 후 후보 세트(원본 clean의 부분집합; index 일치)
        mean/p95/uncert: regressor 예측 벡터(원본 clean 인덱스 기준)
        mode: "hybrid" 또는 "pure-ml" 등
        use_opsigs: 하이브리드에서만 운영 시그널(opsigs) 포함 여부
        """
        # ---------- 공통 예측 벡터 추출 ----------
        idx   = dfK.index.to_numpy()
        mean_ = np.asarray(mean[idx],  dtype=np.float32)
        p95_  = np.asarray(p95[idx],   dtype=np.float32)
        unct_ = np.asarray(uncert[idx], dtype=np.float32)

        # ---------- 공통 파생량 ----------
        # 결측-safe 추출
        est_transfer = dfK.get('est_transfer_ms', 0.0)
        est_jitter   = dfK.get('est_jitter_ms',   0.0)
        p_cold       = dfK.get('p_cold_est',      0.0)
        est_cold_ms  = dfK.get('est_cold_ms',     0.0)

        est_transfer = np.asarray(est_transfer, dtype=np.float32)
        est_jitter   = np.asarray(est_jitter,   dtype=np.float32)
        p_cold       = np.asarray(p_cold,       dtype=np.float32)
        est_cold_ms  = np.asarray(est_cold_ms,  dtype=np.float32)

        # 총 추정 지연(평균 기반) & 꼬리 마진
        est_total = est_transfer + est_jitter + mean_ + p_cold * est_cold_ms
        tail      = np.maximum(0.0, p95_ - mean_).astype(np.float32, copy=False)

        # req별 최소 est_total 대비 margin (delta_to_min) — 원본 로직 유지
        req        = dfK['req_id'].to_numpy(np.int64, copy=False)
        order      = np.argsort(req, kind='mergesort')
        req_sorted = req[order]
        est_sorted = est_total[order]

        new_seg         = np.empty_like(req_sorted, dtype=bool)
        new_seg[0]      = True
        new_seg[1:]     = (req_sorted[1:] != req_sorted[:-1])
        starts          = np.flatnonzero(new_seg)
        ends            = np.r_[starts[1:], len(req_sorted)]
        delta_sorted    = np.empty_like(est_sorted, dtype=np.float32)
        for s, e in zip(starts, ends):
            m = est_sorted[s:e].min()
            delta_sorted[s:e] = est_sorted[s:e] - m
        delta_to_min = np.empty_like(delta_sorted, dtype=np.float32)
        delta_to_min[order] = delta_sorted

        # ---------- opsigs 준비 (필요 시에만 계산) ----------
        # 하이브리드에서만 사용되는 운영 시그널들 (결측-safe + 정규화 가드)
        qdepth = dfK.get('queue_depth', np.zeros(len(dfK), np.float32))
        qdepth = np.asarray(qdepth, dtype=np.float32)
        if qdepth.size:
            qd_min = float(np.nanpercentile(qdepth, 5))
            qd_max = float(np.nanpercentile(qdepth,95))
        else:
            qd_min, qd_max = 0.0, 1.0
        qdepth_norm = (qdepth - qd_min) / (qd_max - qd_min + 1e-9)
        qdepth_norm = np.clip(qdepth_norm, 0.0, 1.5).astype(np.float32, copy=False)

        # 이진 플래그(0/1)
        cache_p    = np.asarray(dfK.get('cache_hit_prob_est', 0.0), dtype=np.float32)
        p_cold_est = np.asarray(dfK.get('p_cold_est',         0.0), dtype=np.float32)
        cache_hit  = (cache_p >= 0.5).astype(np.float32)
        cold_flag  = (p_cold_est >= 0.5).astype(np.float32)

        # 기타 피처(결측-safe)
        is_edge = np.asarray(dfK.get('is_edge', 0.0), dtype=np.float32)
        rho     = np.asarray(dfK.get('rho_proxy', 0.0), dtype=np.float32)
        overm   = np.asarray(dfK.get('over_mem_pos', np.zeros(len(dfK), np.float32)), dtype=np.float32)

        # RTT 근사치 프록시
        rtt_proxy = (est_transfer + est_jitter).astype(np.float32, copy=False)

        # ---------- 피처셋 구성 분기 ----------
        use_opsigs = bool(use_opsigs and (mode == "hybrid"))
        if use_opsigs:
            # Hybrid-ML: opsigs + ml
            feats = np.column_stack([
                -(est_total.astype(np.float32, copy=False)),  # anchor(정렬 안정화)
                mean_.astype(np.float32, copy=False),
                tail.astype(np.float32, copy=False),
                unct_.astype(np.float32, copy=False),

                qdepth_norm,          # 혼잡도
                is_edge,              # 엣지 여부
                rtt_proxy,            # RTT 대리값(전송+지터)
                cache_hit,            # 캐시 히트 플래그
                cold_flag,            # 콜드스타트 플래그
                est_jitter.astype(np.float32, copy=False),
                rho,
                overm,
                delta_to_min,
            ]).astype(np.float32, copy=False)

            self.last_feat_names_ = [
                'neg_est_total','pred_mean','pred_tail','pred_uncert',
                'qdepth_norm','is_edge','rtt_proxy','cache_hit','cold_start',
                'est_jitter_ms','rho_proxy','over_mem_pos','delta_to_min'
            ]
        else:
            # Pure-ML 등: 최소 코어 4피처(논문/리그레서 일관성)
            feats = np.column_stack([
                mean_.astype(np.float32, copy=False),
                tail.astype(np.float32, copy=False),
                unct_.astype(np.float32, copy=False),
                delta_to_min,
            ]).astype(np.float32, copy=False)

            self.last_feat_names_ = ['pred_mean','pred_tail','pred_uncert','delta_to_min']

        return feats
