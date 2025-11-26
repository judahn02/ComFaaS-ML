# Reranker.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import amp
from torch.utils.data import DataLoader, TensorDataset
from Print import log

def _ensure_writable_c(a, dtype):
    if isinstance(a, np.ndarray):
        if (not a.flags.writeable) or (not a.flags['C_CONTIGUOUS']) or a.dtype != np.dtype(dtype):
            return np.array(a, dtype=dtype, order='C', copy=True)
    return a

class ListNetReranker(nn.Module):

    """
    ListNet(옵션: pairwise) 기반 리랭커.

    - 목적: 프루닝된 후보(K per req)에 대해 req 그룹 내 상대적 순위를 학습해 최종 선택 품질을 높임.
    - 데이터 경로:
      · (기본) CUDA 사용 시 학습 텐서를 한 번만 GPU로 올린 뒤, 매 에폭 GPU 인덱싱으로 미니배치 구성(가장 빠름).
      · (대안) 메모리 부족/CPU 강제 시 DataLoader 경로(pinned memory, multi-worker) 사용.
    - 손실: ListNet(listwise cross-entropy) + α·Pairwise(옵션, hinge형 갭 가중).
      · 배치 내부에서 group id로 정렬 후 unique_consecutive 세그먼트로 그룹별 softmax/쌍대 손실 계산.
    - 로깅: v27.3 스타일 에폭 로그(1, 3의 배수, 마지막 에폭) 출력.
    - 메모리/주의: GPU 상주 경로는 전체 학습셋을 GPU에 보유해야 함. 부족 시 force_cpu_loader=True 사용.
    - API: fit(X, y, group, batch), score(X). (X: 특징, y: 음수 지연/유틸리티, group: req_id)

    ---
    English:

    ListNet (optional pairwise) reranker.

    - Goal: Learn within-request ranking over pruned K candidates to improve final selection quality.
    - Data path:
      · (Default) With CUDA, move tensors to GPU once and form mini-batches via GPU indexing each epoch (fastest).
      · (Fallback) If memory is tight or CPU is forced, use a DataLoader path (pinned memory, multi-worker).
    - Losses: ListNet (listwise cross-entropy) + α·Pairwise (optional, hinge-like with gap emphasis).
      · Sort by group id within each batch, then use unique_consecutive segments for per-group softmax/pairwise losses.
    - Logging: v27.3-style epoch logs (at 1, every 3 epochs, and the last).
    - Memory/Notes: The GPU-resident path requires keeping the full training set on GPU. Use force_cpu_loader=True if needed.
    - API: fit(X, y, group, batch), score(X). (X: features, y: negative latency/utility, group: req_id)
    """

    def __init__(self,
                 in_dim,
                 hidden=96,
                 p=0.1,
                 lr=3e-4,
                 lw=1.0,
                 pw=0.0,
                 epochs=15,
                 device=None,
                 force_cpu_loader: bool=False,
                 # --- 추가 하이퍼파라미터 ---
                 pair_topm: int = 4,       # 그룹 내 타깃 상위 m(양성)만 사용
                 pair_negs: int = 16,      # 양성당 음성 샘플 개수
                 pair_start_ep: int = 3,   # 이 에폭 전까지는 listwise만
                 use_amp: bool = True      # AMP로 matmul/activation 가속
                 ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(p),
            nn.Linear(hidden, 1)
        )
        self.lw, self.pw, self.epochs = float(lw), float(pw), int(epochs)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.force_cpu_loader = bool(force_cpu_loader)

        # 추가 pairwise 샘플링 제어
        self.pair_topm = int(pair_topm)
        self.pair_negs = int(pair_negs)
        self.pair_start_ep = int(pair_start_ep)

        # AMP 설정 (cuda에서만 활성)
        self.use_amp = bool(use_amp and (self.device == 'cuda'))

        # 옵티마이저(기존 유지: AdamW)
        self.opt = torch.optim.AdamW(self.parameters(), lr=lr)
        self.to(self.device)

        # AMP GradScaler
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scaler = amp.GradScaler('cuda', enabled=self.use_amp)

        # 성능 튜닝
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # 경로 로그
        log(f"[reranker] device={self.device} | amp={self.use_amp} | "
            f"pair_topm={self.pair_topm} | pair_negs={self.pair_negs} | "
            f"pair_start_ep={self.pair_start_ep} | lw={self.lw} | pw={self.pw} | "
            f"force_cpu_loader={self.force_cpu_loader}")

    # --------------------------- ListNet (벡터화) ---------------------------
    def _listnet_loss_vectorized(self, s, t, g):
        """
        ListNet(listwise cross-entropy) — 벡터화 버전.
        배치 내에서 그룹이 섞여 있어도 됨. 정렬 불필요.

        s: scores [N], t: targets [N], g: group ids [N] (int)
        """
        eps = 1e-9
        uniq, inv = torch.unique(g, return_inverse=True)
        G = uniq.size(0)

        # softmax(s) per group
        exp_s = torch.exp(s)                               # [N]
        den_s = torch.zeros(G, device=s.device)            # [G]
        den_s.index_add_(0, inv, exp_s)                    # sum exp(s) within group
        p = exp_s / (den_s[inv] + eps)                     # [N]

        # softmax(t) per group
        exp_t = torch.exp(t)
        den_t = torch.zeros(G, device=t.device)
        den_t.index_add_(0, inv, exp_t)
        q = exp_t / (den_t[inv] + eps)

        # cross-entropy per element, then sum per group → mean over groups
        ce_elem = -q * torch.log(p + eps)                  # [N]
        ce_group = torch.zeros(G, device=s.device)
        ce_group.index_add_(0, inv, ce_elem)
        return ce_group.mean()

    # ---------------------- Pairwise 완전 벡터화 (그룹 루프 제거) -----------------------
    def _pair_loss_sampled_fast(self, scores, targets, group_ids):
        """
        벡터화 샘플링형 pairwise:
        - 입력은 '그룹 정렬된 배치'(fit에서 보장)
        - 그룹마다 후보 수 ≤ K (프루닝 Kmax). 부족하면 패딩.
        """
        device = scores.device
        # 그룹 경계
        dif = torch.empty_like(group_ids, dtype=torch.bool)
        dif[0] = True
        dif[1:] = group_ids[1:] != group_ids[:-1]
        starts = torch.nonzero(dif, as_tuple=False).flatten()
        ends = torch.cat([starts[1:], torch.tensor([group_ids.numel()], device=device)])
        G = starts.numel()

        # Kmax 추정(그룹 최대 길이)
        K = int(torch.max(ends - starts).item())
        # 텐서 준비
        S = scores.new_full((G, K), float('-inf'))  # score pad=-inf
        T = targets.new_full((G, K), float('-inf')) # target pad=-inf
        M = torch.zeros((G, K), dtype=torch.bool, device=device)  # mask(valid)

        for gi in range(G):
            st = starts[gi].item(); en = ends[gi].item()
            n = en - st
            S[gi, :n] = scores[st:en]
            T[gi, :n] = targets[st:en]
            M[gi, :n] = True

        # top-m (타깃 기준) 인덱스 [G, m]
        m = min(self.pair_topm, K)
        top_vals, top_idx = torch.topk(T, k=m, dim=1, largest=True, sorted=False)
        # 유효 토큰만 남기기
        top_mask = torch.gather(M, 1, top_idx)  # [G, m]
        # pos scores [G, m]
        pos = torch.gather(S, 1, top_idx)
        pos = torch.where(top_mask, pos, torch.full_like(pos, float('-inf')))

        # neg 풀: M & ~one_hot(top_idx)
        one_hot = torch.zeros_like(M, dtype=torch.bool)
        one_hot.scatter_(1, top_idx.clamp(min=0), True)
        neg_pool_mask = M & (~one_hot)

        # 각 그룹에서 neg k개 샘플 인덱스 생성 (복원추출)
        k = min(self.pair_negs, K)
        # 랜덤 인덱스 [G, m, k] in [0, K)
        rand_idx = torch.randint(0, K, (G, m, k), device=device)
        # 유효한 neg만 선택 (mask로 걸러내고, 무효는 -inf가 되도록)
        neg_mask = torch.gather(neg_pool_mask.unsqueeze(1).expand(-1, m, -1), 2, rand_idx)
        neg = torch.gather(S.unsqueeze(1).expand(-1, m, -1), 2, rand_idx)
        neg = torch.where(neg_mask, neg, torch.full_like(neg, float('-inf')))

        # pos [G, m] → [G, m, 1] 브로드캐스트
        diff = pos.unsqueeze(-1) - neg  # [G, m, k]
        # -log σ(diff) = softplus(-diff)
        # 무효(-inf) 항은 0 기여가 되도록 clamp
        diff = torch.nan_to_num(diff, neginf=-1e6)  # 큰 음수 → σ≈0 → loss≈~0
        loss = F.softplus(-diff)
        # 유효 마스크
        val_mask = top_mask.unsqueeze(-1) & neg_mask
        loss = (loss * val_mask).sum() / (val_mask.sum().clamp(min=1))
        return loss

    # (선택) 레거시 빠른 정렬형 pairwise — 유지만 해둠(기본 경로는 sampled 사용)
    def _pair_loss_sorted_fast(self, s, t, g):
        """
        Pairwise hinge-like loss (expects g sorted & contiguous).
        For each group, compare best (by t) vs all others:
            loss_seg = sum_j ReLU(s_j - s_best) * (1 + 3*ReLU(t_best - t_j))
        """
        uniq, counts = torch.unique_consecutive(g, return_counts=True)
        start = 0
        loss = s.new_zeros(())
        one = s.new_ones(())
        three = s.new_tensor(3.0)

        for c in counts.tolist():
            seg = slice(start, start + c)
            s_seg = s[seg]
            t_seg = t[seg]

            b = torch.argmax(t_seg)
            s_best = s_seg[b]
            t_best = t_seg[b]

            hinge = torch.relu(s_seg - s_best)
            w = one + three * torch.relu(t_best - t_seg)
            if c > 1:
                loss = loss + torch.sum(hinge * w)
            start += c
        return loss / max(1, len(uniq))

    # ------------------------------- 학습 -------------------------------
    def fit(self, X, y, group, batch=8192):

        """
        Reranker 학습 루프.
        - GPU 상주 경로: 입력 전체를 GPU에 올린 뒤, 한 번만 '요청(그룹) 기준으로 정렬'하고
        에폭마다 '그룹 인덱스만 셔플'하여 배치를 구성한다.
            * 효과: 같은 요청의 후보 K(≤Kmax)가 동일 배치에 모여 pairwise 루프 호출이 대폭 감소.
            * ListNet(listwise) 벡터화 + (ep ≥ pair_start_ep)에서 샘플링형 pairwise 손실을 추가.
            * AMP(autocast + GradScaler)로 matmul/activation 가속 및 안정적 스케일링.
        - CPU 경로: DataLoader(pinned memory, multi-worker)로 일반 미니배치 학습.

        Args:
            X (ndarray, shape [N, D]): 프루닝 이후 리랭커 입력 피처.
            y (ndarray, shape [N]):   그룹 내 상대 순위를 학습하기 위한 타깃
                                    (파이프라인에서 req별 z-score된 -total_latency_ms_realized).
            group (ndarray, shape [N]): 요청(그룹) ID. 같은 요청의 후보가 동일 ID를 갖는다.
            batch (int): 미니배치 크기(권장: 16,384~65,536; GPU 여유에 따라 조정).

        Loss:
            total = lw * ListNet(s, y, g) + pw * Pairwise(s, y, g)
            - ListNet: 완전 벡터화된 group-wise cross-entropy.
            - Pairwise(옵션): 타깃 상위 m(양성) × 음성 k개 샘플(복원추출)만 사용(샘플링형).
            ep < pair_start_ep 동안은 꺼서 초기 수렴 가속.
            (복잡도 O(G * m * k)로, 모든 쌍 O(n^2) 대비 크게 경감.)

        처리 흐름( GPU 경로 ):
            1) X,y,group를 GPU에 상주시킨 뒤 group 기준 정렬(order) 및 그룹 경계(starts/ends) 계산(1회).
            2) 각 에폭에서 '그룹 인덱스'만 랜덤 셔플 후, 여러 그룹을 이어 붙여 배치 구성.
            3) autocast → forward → ListNet(+optional pairwise) → backward(GradScaler) → step.
            4) ep ∈ {1, 3의 배수, 마지막}에 요약 로깅.

        Notes:
            - pairwise_w>0인 경우에도 '그룹 정렬 배치'를 통해 루프 호출이 크게 줄어 학습시간을 안정화.
            - AMP가 켜져 있으면 fp16 경로에서 matmul/activation이 가속된다.
            - grad clipping(5.0)으로 드문 스파이크를 방지.
            - CPU 경로는 메모리 제약/강제 시 자동 사용되며, pinned memory + prefetch로 I/O 오버헤드를 완화.

        Returns:
            self (ListNetReranker): 학습된 모델(메서드 체이닝 가능).
        """

        use_gpu_dataset = (self.device == 'cuda') and (not self.force_cpu_loader)

        # 경로/스펙 로깅
        try:
            dim = X.shape[1]
        except Exception:
            dim = "NA"
        log(f"[reranker] device={self.device} | force_cpu_loader={self.force_cpu_loader} | gpu_path={use_gpu_dataset}")
        log(f"[reranker] N={len(X)} | dim={dim} | epochs={self.epochs} | batch={batch}")
        # fit() 시작부: 데이터 크기/예상 스텝 로그
        try:
            import math
            n = len(X)
            exp_steps = math.ceil(n / max(1, int(batch)))
            log(f"[reranker.fit] expected steps/epoch ≈ {exp_steps} (N={n}, batch={int(batch)})")
        except Exception:
            pass

        # =========================
        #   GPU 상주 경로 (개선)
        # =========================
        if use_gpu_dataset:
            # 0) 텐서 상주시 + 정렬 준비
            # X가 read-only numpy면 안전하게 writable 복사본으로 만들어 준다 (경고/UB 방지)
            X = _ensure_writable_c(X, np.float32)
            y = _ensure_writable_c(y, np.float32)
            group = _ensure_writable_c(group, np.int64)

            Xg = torch.as_tensor(X, dtype=torch.float32, device=self.device).contiguous()
            yg = torch.as_tensor(y, dtype=torch.float32, device=self.device).contiguous()
            gg = torch.as_tensor(group, dtype=torch.long,   device=self.device).contiguous()


            # -- 핵심: 그룹 정렬 인덱스(한 번만 계산)
            order = torch.argsort(gg, stable=True)
            g_sorted = gg[order]
            X_sorted = Xg[order]; y_sorted = yg[order]

            # -- 그룹 경계(한 번만 계산)
            dif = torch.empty_like(g_sorted, dtype=torch.bool)
            dif[0] = True
            dif[1:] = (g_sorted[1:] != g_sorted[:-1])
            starts = torch.nonzero(dif, as_tuple=False).flatten()
            ends   = torch.cat([starts[1:], torch.tensor([g_sorted.numel()], device=self.device)])
            G = starts.numel()
            group_ids = torch.arange(G, device=self.device)

            # GPU 경로: 실제 스텝 수 계측
            for ep in range(1, self.epochs + 1):
                #self.train(); tot = 0.0; n = 0
                self.train(); tot = 0.0; n = 0; step_count = 0

                # 1) "그룹 인덱스"만 셔플 → 그룹 단위로 배치 구성
                perm_groups = group_ids[torch.randperm(G, device=self.device)]

                buf_idx = []
                # 2) 그룹들을 순서대로 쌓아가며, 버퍼가 batch 이상이 되면 한 번에 학습
                for gi in perm_groups.tolist():
                    st = starts[gi].item(); en = ends[gi].item()
                    # 이 그룹(요청)의 모든 후보를 통째로 버퍼에 쌓는다
                    buf_idx.extend(range(st, en))

                    if len(buf_idx) >= batch:
                        idx = torch.as_tensor(buf_idx, device=self.device)
                        xb = X_sorted.index_select(0, idx)
                        yb = y_sorted.index_select(0, idx)
                        gb = g_sorted.index_select(0, idx)

                        self.opt.zero_grad(set_to_none=True)
                        with amp.autocast('cuda', enabled=self.use_amp):
                            scores = self.net(xb).squeeze(-1)
                            loss = self.lw * self._listnet_loss_vectorized(scores, yb, gb)
                            if self.pw > 0.0 and ep >= self.pair_start_ep:
                                # 샘플링형 pairwise (정렬 필요 없음 / 그룹 경계는 배치 구성으로 보장)
                                loss = loss + self.pw * self._pair_loss_sampled_fast(scores, yb, gb)

                        self.scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                        self.scaler.step(self.opt); self.scaler.update()
                        step_count += 1

                        tot += loss.item() * xb.size(0); n += xb.size(0)
                        buf_idx.clear()

                # 3) 남은 버퍼 flush
                if buf_idx:
                    idx = torch.as_tensor(buf_idx, device=self.device)
                    xb = X_sorted.index_select(0, idx)
                    yb = y_sorted.index_select(0, idx)
                    gb = g_sorted.index_select(0, idx)

                    self.opt.zero_grad(set_to_none=True)
                    with amp.autocast('cuda', enabled=self.use_amp):
                        scores = self.net(xb).squeeze(-1)
                        loss = self.lw * self._listnet_loss_vectorized(scores, yb, gb)
                        if self.pw > 0.0 and ep >= self.pair_start_ep:
                            loss = loss + self.pw * self._pair_loss_sampled_fast(scores, yb, gb)

                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                    
                    self.scaler.step(self.opt); self.scaler.update()
                    step_count += 1

                    tot += loss.item() * xb.size(0); n += xb.size(0)

                if ep == 1 or ep % 3 == 0 or ep == self.epochs:
                    # log(f"[reranker] epoch {ep:4d}/{self.epochs} | total {tot/max(1,n):.4f}")
                    log(f"[reranker] epoch {ep:4d}/{self.epochs} | total {tot/max(1,n):.4f} | steps={step_count}")

            return self

        # =========================
        #   CPU DataLoader 경로
        # =========================
        Xc = torch.as_tensor(X, dtype=torch.float32, device='cpu').contiguous()
        yc = torch.as_tensor(y, dtype=torch.float32, device='cpu').contiguous()
        gc = torch.as_tensor(group, dtype=torch.long,   device='cpu').contiguous()

        pin = (self.device == 'cuda')
        cpu_cnt = os.cpu_count() or 4
        nw = max(1, min(8, cpu_cnt // 2))
        dl = DataLoader(
            TensorDataset(Xc, yc, gc),
            batch_size=batch, shuffle=True,
            pin_memory=pin, num_workers=nw,
            persistent_workers=True, prefetch_factor=4
        )

        for ep in range(1, self.epochs + 1):
            self.train(); tot = 0.0; n = 0
            for xb_c, yb_c, gb_c in dl:
                xb = xb_c.to(self.device, non_blocking=pin)
                yb = yb_c.to(self.device, non_blocking=pin)
                gb = gb_c.to(self.device, non_blocking=pin)

                self.opt.zero_grad(set_to_none=True)
                with amp.autocast('cuda', enabled=self.use_amp):
                    scores = self.net(xb).squeeze(-1)
                    loss = self.lw * self._listnet_loss_vectorized(scores, yb, gb)
                    if self.pw > 0.0 and ep >= self.pair_start_ep:
                        loss = loss + self.pw * self._pair_loss_sampled_fast(scores, yb, gb)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                self.scaler.step(self.opt); self.scaler.update()
                step_count += 1

                tot += loss.item() * xb.size(0); n += xb.size(0)

            if ep == 1 or ep % 3 == 0 or ep == self.epochs:
                # log(f"[reranker] epoch {ep:4d}/{self.epochs} | total {tot/max(1,n):.4f}")
                log(f"[reranker] epoch {ep:4d}/{self.epochs} | total {tot/max(1,n):.4f} | steps={step_count}")
    
        return self


    # ------------------------------- 추론 -------------------------------
    def score(self, X):
        self.eval()
        with torch.no_grad():
            Xd = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self.net(Xd).squeeze(-1).cpu().numpy()
