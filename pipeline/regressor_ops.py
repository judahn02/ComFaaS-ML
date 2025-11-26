# pipeline/regressor_ops.py

# stdlib
import time

# import numpy as np, pandas as pd
# from typing import Tuple, Optional

# # third-party
# from sklearn.preprocessing import StandardScaler

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# project-local
from Features import RegressorBlock
from Data import build_proxy_targets
from .common import DEVICE


# 공통 유틸
from .common import log, _force_determinism, _values_for_sub, _round2, _jsonify

# (필요하면) 외부 모듈 임포트도 원래대로 추가: e.g., from Regressor import MLPRegressor
from Regressor import MLPRegressor

# === A) Regressor train ======================================================
def _hyb_reg_train(args, clean: pd.DataFrame) -> Tuple["MLPRegressor", "StandardScaler"]:
    """[A) Regressor train] 3) Regressor train (fit + scaler.fit_transform) 블록 전체를 이식.
    in : args, clean
    out: (reg, scaler)
    """
    # 3) Regressor train (fit + scaler.fit_transform) -------------------------
    X_raw = RegressorBlock().build(clean)
    y_mean, y_p95 = build_proxy_targets(clean)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    reg = MLPRegressor(
        in_dim=X.shape[1], hidden=args.reg_hidden, p=args.reg_dropout,
        lr=args.reg_lr, weight_decay=args.weight_decay,
        quantile_w=args.quantile_weight, device=DEVICE
    )

    t_train_s = time.time()
    reg.fit(X, y_mean, y_p95, epochs=args.reg_epochs, batch=args.batch)
    t_train_e = time.time()
    log(f"[Regressor] train Wall time: {t_train_e - t_train_s:.2f} s")

    return reg, scaler

# === B) Regressor infer ======================================================
def _hyb_reg_infer(reg: "MLPRegressor",
                   scaler: "StandardScaler",
                   clean: pd.DataFrame,
                   args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[B) Regressor infer] 4) Regressor infer (scaler.transform 재사용) 블록 전체를 이식.
    in : reg, scaler, clean, args
    out: (mean, p95, uncert)
    """
    # 4) Regressor infer (reuse scaler.transform) -----------------------------
    X_full = RegressorBlock().build(clean)
    X_full = scaler.transform(X_full)
    mean, p95, uncert = reg.predict(X_full, T=args.mc_samples)

    try:
        d = (p95 - mean).astype(float)
        log(f"[sanity] mean(p95-mean) = {float(np.nanmean(d)):.4f} ms | share(<=0) = {float(np.mean(d<=0))*100:.2f}%")
    except Exception:
        pass

    return mean, p95, uncert
