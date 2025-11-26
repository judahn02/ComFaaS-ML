# pipeline/common.py

import os, json, time
import numpy as np
import pandas as pd

# 기존 프로젝트의 로깅/저장 유틸은 Print 모듈에 있음
from Print import log, save_df, save_json

# device autodetect (shared)
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# --- reporting version (deterministic baseline v2) ---
META_VERSION = "ComFaaS-ML 70.1.1-deterministic"

# --- one-time CFG summary emitter -------------------------------------------
import os as _os
_CFG_SUMMARY_EMITTED = False

def emit_cfg_once(extra: dict):
    """
    Print one-line config summary only once per process,
    even if this module is imported via multiple aliases.
    """
    global _CFG_SUMMARY_EMITTED
    if _CFG_SUMMARY_EMITTED:
        return
    if _os.environ.get("COMFAAS_CFG_EMITTED") == "1":
        _CFG_SUMMARY_EMITTED = True
        return

    line = (
        f"[CFG] priors_used={str(extra.get('priors_used', False)).lower()} | "
        f"tie_policy={extra.get('tie_policy', 'neutral')} | "
        f"rr_featset={extra.get('rr_featset', 'ml-only')}"
    )
    try:
        from Print import log
        log(line)
    except Exception:
        print(line)

    _CFG_SUMMARY_EMITTED = True
    _os.environ["COMFAAS_CFG_EMITTED"] = "1"
# ---------------------------------------------------------------------------


# --- Determinism helper (for reranker/regressor training) ---
def _force_determinism(seed: int):
    import os, random
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import torch as _torch
    except Exception:
        _torch = None

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed)
    if _torch is not None:
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        try:
            _torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
        except Exception:
            pass

def _values_for_sub(x, sub_index: pd.Index) -> np.ndarray:
    """
    Return 1D float ndarray aligned to sub_index from x (Series or ndarray).
    - If x is a Series: use label-based .loc on sub_index.
    - If x is an ndarray: try positional take with sub_index if possible.
      If lengths already match sub length, return as-is.
      Scalars broadcast to sub length.
    """
    if isinstance(x, pd.Series):
        out = x.loc[sub_index].to_numpy(dtype=float)
        return out

    arr = np.asarray(x)
    if arr.ndim == 0:
        return np.full(len(sub_index), float(arr), dtype=float)

    arr = arr.astype(float, copy=False)
    try:
        if len(arr) >= (int(sub_index.max()) + 1):
            return arr[sub_index]
    except Exception:
        pass

    if arr.shape[0] == len(sub_index):
        return arr

    if arr.shape[0] > len(sub_index):
        return arr[:len(sub_index)]
    out = np.empty(len(sub_index), dtype=float)
    out[: arr.shape[0]] = arr
    out[arr.shape[0]:] = np.nan
    return out

# 70.1.1 optional: human-friendly rounding for JSON report
def _round2(x):
    try:
        return round(float(x), 2)
    except Exception:
        return x

# 70.1.1 optional: JSON-safe casting (handles numpy scalars, lists/tuples/dicts)
def _jsonify(o):
    try:
        import numpy as _np
    except Exception:
        _np = None

    # 기본 스칼라형
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o

    # numpy 스칼라형
    if _np is not None:
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.bool_):
            return bool(o)

    # 컨테이너형
    if isinstance(o, (list, tuple)):
        return [_jsonify(x) for x in o]
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}

    # 그 외(예: enum, pathlib 등)는 문자열로 안전 변환
    return str(o)

# --- stage wall-time timer ---------------------------------------------------
import time
from contextlib import contextmanager

@contextmanager
def stage_timer(name: str, sync_cuda: bool = True):
    # CUDA 동기화는 선택(훈련/스코어 측정 정확도 ↑, 아주 미세한 오버헤드)
    try:
        if sync_cuda and DEVICE == "cuda":
            import torch  # local import: torch 미존재 환경 대비
            if hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize()
    except Exception:
        pass

    t0 = time.perf_counter()
    try:
        yield
    finally:
        try:
            if sync_cuda and DEVICE == "cuda":
                import torch
                if hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
                    torch.cuda.synchronize()
        except Exception:
            pass
        dt = time.perf_counter() - t0
        log(f"[time] {name}: {dt:.2f}s")
# ---------------------------------------------------------------------------
