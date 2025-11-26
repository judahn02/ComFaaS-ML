# pipeline/rerank_stage.py

# stdlib
import time, sys

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch

from Reranker import ListNetReranker
from Features import RerankerBlock

# project-local
from .common import log, DEVICE, _force_determinism, _values_for_sub

# --- Feature cache for fast_reranker (module-level) ---
# _FEATURE_CACHE = {}
_FEATURE_CACHE: Dict[Tuple, object] = {} # <-- 타입 힌트 추가

# === D) Train reranker (optional) ===========================================
def _hyb_train_reranker_if_needed(pruned: pd.DataFrame,
                                  realized: pd.DataFrame,
                                  mean: np.ndarray,
                                  p95: np.ndarray,
                                  uncert: np.ndarray,
                                  args,
                                  mode: str,
                                  train_rr_override: Optional[bool]) -> Optional["ListNetReranker"]:
    """[D) Train reranker] 7) reranker 학습(그룹표준화 포함). hybrid는 플래그, pure-ml은 강제 ON.
    in : pruned, realized, mean, p95, uncert, args, mode, train_rr_override
    out: reranker or None
    """
    # 7) Train reranker (optional; group-standardized targets) ----------------
    tr0 = time.time()
    reranker = None

    # pure-ml 모드는 reranker 강제 ON, hybrid는 플래그 따름
    train_rr = getattr(args, "train_reranker", False)
    if mode == "pure-ml":
        train_rr = True
    if train_rr_override is not None:
        train_rr = bool(train_rr_override)

    if train_rr:
        _force_determinism(int(getattr(args, "seed", 0)) + 19)

        # ==== 특징 생성 (캐시 적용 지점) ====
        seed   = int(getattr(args, "seed", 0))
        n_rows = int(pruned.shape[0])

        # 인덱스 요약(충돌 완화용)
        if n_rows > 0:
            try:
                idx0 = int(pruned.index[0])
                idx1 = int(pruned.index[-1])
            except Exception:
                idx0 = str(pruned.index[0])
                idx1 = str(pruned.index[-1])
        else:
            idx0 = idx1 = -1

        # req_id/node_id 간단 통계(있으면)
        try:
            req_arr = pruned["req_id"].to_numpy()
            nid_arr = pruned["node_id"].to_numpy()
            req_sum = int(req_arr.sum()); nid_sum = int(nid_arr.sum())
            req_max = int(req_arr.max()); nid_max = int(nid_arr.max())
        except Exception:
            req_sum = nid_sum = req_max = nid_max = 0

        # ----------------- (추가) 하이브리드 opsigs 분기 -----------------
        # 하이브리드에서만 의미 있는 운영 시그널 사용 여부
        use_opsigs = bool(
            (mode == "hybrid")
            and getattr(args, "reranker_use_opsigs", False)
            and not getattr(args, "reranker_opsigs_off", False)
        )
        # feat 스키마가 달라지므로 캐시 키에 반드시 포함
        featset_tag = ("opsigs+ml" if use_opsigs else "ml-only")
        # ← report용 임시 보관(최소침습)
        try:
            setattr(args, "_report_rr_featset", featset_tag)
        except Exception:
            pass

        # 버전 태그 포함 키(포맷 바꾸면 접미사 증분: v1 -> v2)
        cache_key = ("Xk_v2", seed, n_rows, idx0, idx1, req_sum, nid_sum, req_max, nid_max, mode, featset_tag)

        use_cache = bool(getattr(args, "fast_reranker", False)) and n_rows > 0

        if use_cache and cache_key in _FEATURE_CACHE:
            Xk = _FEATURE_CACHE[cache_key]
            log(f"[reranker] feature cache HIT (n={n_rows}, mode={mode}, featset={featset_tag})")

            # ▼ 읽기 전용 가드 (numpy / torch 모두 커버, 예외시 무시)
            try:
                if isinstance(Xk, np.ndarray):
                    Xk.setflags(write=False)
            except Exception:
                pass
            try:
                if torch.is_tensor(Xk):
                    Xk.requires_grad_(False)
            except Exception:
                pass
        else:
            # ★ 여기서 모드/opsigs를 명시적으로 전달
            Xk = RerankerBlock().build(
                pruned,
                mean[pruned.index], p95[pruned.index], uncert[pruned.index],
                mode=mode, use_opsigs=use_opsigs
            )
            log(f"[reranker] feature build (n={n_rows}, mode={mode}, featset={featset_tag}, in_dim={Xk.shape[1]})")

            if use_cache:
                # 단순 LRU(1개 보관): 메모리 보호. 다회 실험 아니면 이게 가장 안전.
                _FEATURE_CACHE.clear()
                # ▼ 저장 전에 read-only 세팅
                try:
                    if isinstance(Xk, np.ndarray):
                        Xk.setflags(write=False)
                except Exception:
                    pass
                try:
                    if torch.is_tensor(Xk):
                        Xk.requires_grad_(False)
                except Exception:
                    pass
                _FEATURE_CACHE[cache_key] = Xk
                log(f"[reranker] feature cache SAVE (n={n_rows}, mode={mode}, featset={featset_tag})")
        # ==== 특징 생성 끝 ====

        # merge+sort → dict-map 대체 (fast일 때만)
        if getattr(args, "fast_reranker", False):
            key_r = list(zip(realized['req_id'].to_numpy(), realized['node_id'].to_numpy()))
            val_r = realized['total_latency_ms_realized'].to_numpy(np.float32)
            mp = dict(zip(key_r, val_r))
            y_ms = -np.array([ mp.get(k, np.nan) for k in zip(pruned['req_id'], pruned['node_id']) ], dtype=np.float32)
        else:
            joined = pruned.merge(
                realized[['req_id','node_id','total_latency_ms_realized']],
                on=['req_id','node_id'], how='left'
            ).sort_values(['req_id','node_id'], kind='mergesort')
            y_ms = -joined['total_latency_ms_realized'].to_numpy(np.float32)

        # 벡터화 (fast일 때만)
        req_ids = pruned['req_id'].to_numpy(np.int64)
        y_ms    = y_ms.astype(np.float32, copy=False)

        if getattr(args, "fast_reranker", False):
            s = pd.Series(y_ms)
            g = pd.Series(req_ids)
            m = s.groupby(g).transform('mean').to_numpy(np.float32)
            v = s.groupby(g).transform('std').to_numpy(np.float32) + 1e-6
            y_gt = ((y_ms - m) / v).astype(np.float32, copy=False)
        else:
            y_gt = y_ms.copy()
            for rid in np.unique(req_ids):
                sel = (req_ids == rid)
                m = y_gt[sel].mean()
                s_ = y_gt[sel].std() + 1e-6
                y_gt[sel] = (y_gt[sel] - m) / s_

        # -------------------- (추가) 하이퍼 파라미터 로컬화 --------------------
        # 기본값은 기존 파이프라인과 동일하게 유지
        epochs      = int(getattr(args, "rerank_epochs", 15))
        batch       = int(getattr(args, "rerank_batch", 16384)) or 16384  # 기존 고정값 보존
        listwise_w  = float(getattr(args, "listwise_w", 1.0))
        pairwise_w  = float(getattr(args, "pairwise_w", 0.0))

        if getattr(args, "fast_reranker", False):
            # 사용자가 CLI에 --rerank_epochs를 명시했는지 감지
            _argv = sys.argv if isinstance(sys.argv, list) else []
            user_set_epochs = any(
                (a == "--rerank_epochs") or a.startswith("--rerank_epochs=")
                for a in _argv
            )

            # fast 기본 튜닝은 유지
            batch      = max(batch, 32768)
            pairwise_w = 0.0

            if user_set_epochs:
                try:
                    log(f"[reranker] fast mode ON → epochs(respect user)={epochs}, batch={batch}, pairwise_w={pairwise_w:.1f}")
                except Exception:
                    pass
            else:
                epochs = max(5, epochs // 3)
                try:
                    log(f"[reranker] fast mode ON → epochs(adjusted)={epochs}, batch={batch}, pairwise_w={pairwise_w:.1f}")
                except Exception:
                    pass
        # ---------------------------------------------------------------------

        # 3) 모델 생성 시 로컬 변수 사용
        reranker = ListNetReranker(
            in_dim=Xk.shape[1],
            hidden=args.rerank_hidden, p=args.rerank_dropout,
            lr=args.rerank_lr,
            lw=listwise_w, pw=pairwise_w, epochs=epochs, # ← 로컬 변수 전달
            device=DEVICE
        )
        t_fit0 = time.time()
        reranker.fit(Xk, y_gt, pruned['req_id'].to_numpy(np.int64), batch=batch)
        t_fit1 = time.time()
        log(f"[Reranker] fit() time: {t_fit1 - t_fit0:.2f} s (epochs={epochs}, batch={batch})")

    tr1 = time.time()
    if reranker is not None:
        log(f"[Reranker] train Wall time: {tr1 - tr0:.2f} s")

    return reranker
