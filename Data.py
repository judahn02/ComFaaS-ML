# Data.py (ComFaaS_ML_29.1)
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# CSV loader
# ------------------------------------------------------------
class CSVSource:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        # 필수 컬럼이 빠져도 downstream에서 터지지 않도록 최소 백필
        must_have_float = [
            'input_size_kb','est_transfer_ms','est_jitter_ms',
            'p_cold_est','est_cold_ms','queue_depth','rho_proxy',
            'cpu_util_ema','mem_util_ema','cache_hit_prob_est'
        ]
        for c in must_have_float:
            if c not in df.columns:
                df[c] = 0.0

        if 'is_edge' not in df.columns:
            df['is_edge'] = 0
        if 'req_id' not in df.columns:
            # 없으면 0..N//nodes 로 가정해서 채우기 (최소 보정)
            n = len(df)
            if 'node_id' in df.columns:
                # req_id가 없고 node_id가 있으면, 요청당 노드 수 추정
                # 안전하게 1로 둔다.
                df['req_id'] = np.arange(n, dtype=np.int64)
            else:
                df['req_id'] = np.arange(n, dtype=np.int64)
        if 'node_id' not in df.columns:
            # 없으면 0..N-1 (의미 없는 placeholder)
            df['node_id'] = np.arange(len(df), dtype=np.int64)

        # === 새 3축 하드웨어 스코어 백필 ===
        for col, val in [('hardware_cpu', 1.0), ('hardware_gpu', 1.0), ('hardware_storage', 1.0)]:
            if col not in df.columns:
                df[col] = val

        # 선택: is_gpu 컬럼 백필 (없으면 전부 False)
        if 'is_gpu' not in df.columns:
            df['is_gpu'] = 0

        # dtype 정리
        float_cols = [
            'input_size_kb','est_transfer_ms','est_jitter_ms','p_cold_est','est_cold_ms',
            'queue_depth','rho_proxy','cpu_util_ema','mem_util_ema','cache_hit_prob_est',
            'hardware_cpu','hardware_gpu','hardware_storage'
        ]
        for c in float_cols:
            df[c] = df[c].astype('float32', copy=False)
        df['is_edge'] = df['is_edge'].astype('int8', copy=False)
        df['is_gpu']  = df['is_gpu'].astype('int8', copy=False)

        return df


# ------------------------------------------------------------
# Synthetic generator
# ------------------------------------------------------------
class SyntheticSource:
    """
    합성 데이터 생성기.
    - reqs: 요청 개수
    - nodes: 요청당 후보 노드 수
    """
    def __init__(self, reqs: int, nodes: int, seed: int = 42):
        self.reqs = int(reqs)
        self.nodes = int(nodes)
        self.seed = int(seed)

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.reqs * self.nodes

        req_id = np.repeat(np.arange(self.reqs, dtype=np.int64), self.nodes)
        node_id = np.tile(np.arange(self.nodes, dtype=np.int64), self.reqs)

        # 기본 피처 생성
        input_size_kb = rng.lognormal(mean=6.0, sigma=0.75, size=n).astype('float32')  # ~400KB~ 수 MB
        est_transfer_ms = (input_size_kb / 256.0 * rng.uniform(0.15, 0.7, size=n)).astype('float32')
        est_jitter_ms   = rng.uniform(0.1, 4.0, size=n).astype('float32')

        p_cold_est  = rng.beta(1.5, 8.0, size=n).astype('float32')          # cold 확률
        est_cold_ms = rng.uniform(10, 180, size=n).astype('float32')        # cold penalty
        queue_depth = rng.integers(0, 8, size=n).astype('float32')

        rho_proxy       = np.clip(rng.normal(0.65, 0.12, size=n), 0.0, 1.2).astype('float32')
        cpu_util_ema    = np.clip(rng.normal(0.35, 0.18, size=n), 0.0, 1.0).astype('float32')
        mem_util_ema    = np.clip(rng.normal(0.42, 0.20, size=n), 0.0, 1.0).astype('float32')
        cache_hit_prob_est = np.clip(rng.normal(0.75, 0.12, size=n), 0.0, 1.0).astype('float32')

        # 엣지/코어 플래그 (반반)
        is_edge = rng.integers(0, 2, size=n).astype('int8')
        # GPU 유무 (엣지 비율 낮고, 코어 비율 높게 예시)
        is_gpu = np.where(is_edge==1, rng.random(n) < 0.15, rng.random(n) < 0.55).astype('int8')

        # === 3축 하드웨어 스코어 생성 (0.3~2.0) ===
        hardware_cpu     = rng.uniform(0.3, 2.0, size=n).astype('float32')
        hardware_gpu     = rng.uniform(0.3, 2.0, size=n).astype('float32')
        hardware_storage = rng.uniform(0.3, 2.0, size=n).astype('float32')
        # 간단 편향: 엣지는 storage 약간 낮게, 코어는 gpu 약간 낮게 (예시)
        hardware_storage = hardware_storage * np.where(is_edge==1, 0.90, 1.00)
        hardware_gpu     = hardware_gpu     * np.where(is_edge==0, 0.90, 1.00)

        df = pd.DataFrame({
            'req_id': req_id,
            'node_id': node_id,
            'input_size_kb': input_size_kb,
            'est_transfer_ms': est_transfer_ms,
            'est_jitter_ms': est_jitter_ms,
            'p_cold_est': p_cold_est,
            'est_cold_ms': est_cold_ms,
            'queue_depth': queue_depth,
            'rho_proxy': rho_proxy,
            'cpu_util_ema': cpu_util_ema,
            'mem_util_ema': mem_util_ema,
            'cache_hit_prob_est': cache_hit_prob_est,
            'is_edge': is_edge,
            'is_gpu': is_gpu,
            'hardware_cpu': hardware_cpu,
            'hardware_gpu': hardware_gpu,
            'hardware_storage': hardware_storage,
        })

        return df


# ------------------------------------------------------------
# Proxy targets for regressor (mean, p95)
# ------------------------------------------------------------
def build_proxy_targets(df: pd.DataFrame):
    """
    mean / p95 학습용 합성 타깃 생성.
    - Compute 효율: GPU 노드는 hardware_gpu, 그 외는 hardware_cpu 사용 (값이 클수록 빠름)
    - Cold penalty: hardware_storage가 높을수록 감쇠
    - 네트워크/지터는 외생 변수로 별도 처리(여기서는 타깃에 직접 더하지 않음)
    """
    size_kb = df['input_size_kb'].to_numpy('float32')

    if 'is_gpu' in df.columns:
        is_gpu = df['is_gpu'].to_numpy(np.int8, copy=False).astype(bool)
    else:
        is_gpu = np.zeros(len(df), dtype=bool)

    hw_cpu = df['hardware_cpu'].to_numpy('float32') if 'hardware_cpu' in df.columns else np.ones(len(df), dtype='float32')
    hw_gpu = df['hardware_gpu'].to_numpy('float32') if 'hardware_gpu' in df.columns else np.ones(len(df), dtype='float32')
    hw_sto = df['hardware_storage'].to_numpy('float32') if 'hardware_storage' in df.columns else np.ones(len(df), dtype='float32')

    # 연산 효율 (값 클수록 빠름)
    # eff_compute = np.where(is_gpu, hw_gpu, hw_cpu)
    eff_compute = hw_cpu + 0.25 * hw_gpu
    eff_compute = np.clip(eff_compute, 0.3, None)

    # mean proxy: 데이터 크기에 비례, 효율에 반비례
    base_compute = (size_kb / 256.0) * (1.6 / eff_compute)

    # storage 빠를수록 cold penalty 감쇠
    p_cold = df['p_cold_est'].to_numpy('float32')
    cold_ms = df['est_cold_ms'].to_numpy('float32')
    cold_penalty = p_cold * cold_ms * (1.0 / np.clip(hw_sto, 0.3, None))**0.5

    # 최종 타깃
    y_mean = base_compute
    y_p95  = y_mean + 0.20 * y_mean + 0.30 * np.sqrt(y_mean + 1.0) + 0.15 * cold_penalty

    return y_mean.astype('float32'), y_p95.astype('float32')


# ------------------------------------------------------------
# Realized latency for evaluation
# ------------------------------------------------------------
def attach_realized_latency(df: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    """
    평가용 '실측 지연'을 합성해서 붙인다.
    - total_latency_ms_realized = est_transfer + est_jitter + compute + cold + noise
    - compute는 build_proxy_targets의 mean을 기반으로 약간의 잡음을 추가
    """
    rng = np.random.default_rng(seed)

    # compute mean 을 proxy에서 다시 생성
    y_mean, _ = build_proxy_targets(df)

    est_transfer_ms = df['est_transfer_ms'].to_numpy('float32')
    est_jitter_ms   = df['est_jitter_ms'].to_numpy('float32')
    p_cold_est      = df['p_cold_est'].to_numpy('float32')
    est_cold_ms     = df['est_cold_ms'].to_numpy('float32')

    # 약간의 multiplicative / additive 노이즈
    compute_real = y_mean * rng.lognormal(mean=0.0, sigma=0.15, size=len(df)).astype('float32')
    cold_real    = p_cold_est * est_cold_ms * rng.lognormal(mean=0.0, sigma=0.10, size=len(df)).astype('float32')
    net_real     = (est_transfer_ms + est_jitter_ms) * rng.lognormal(mean=0.0, sigma=0.05, size=len(df)).astype('float32')

    total = compute_real + cold_real + net_real
    out = df.copy()
    out['total_latency_ms_realized'] = total.astype('float32')
    return out
