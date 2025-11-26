# Regressor.py
import os, time, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from Print import log

class MLPRegressor(nn.Module):

    """
    Mean + tail head (p95 = mean + softplus(tail)).
    - 기본: CUDA면 학습 텐서를 한 번만 GPU로 올려서 슬라이싱으로 학습(가장 빠름).
    - CPU 강제시에는 DataLoader(pinned memory, multi-worker) 경로 사용.
    - v27.3와 동일한 에폭 로그 + 3줄 요약 출력.

    English:
    - Default: If CUDA is available, move training tensors to GPU once and train via slicing (fastest path).
    - When forced to CPU, fall back to a DataLoader path (pinned memory, multi-worker).
    - Emits the same per-epoch logs and the final 3-line summary as v27.3.
    """

    """
    Mean + tail head (p95 = mean + softplus(tail)).
    CUDA일 때 학습 데이터(X, y_mean, y_p95)를 한 번만 GPU로 올리고,
    에폭마다 GPU 인덱싱으로 미니배치를 구성해 가장 빠르게 학습합니다.
    CPU를 강제해야 하면 DataLoader 경로(pinned memory, multi-worker)를 사용합니다.

    아키텍처: Linear → SiLU → Dropout → Linear → SiLU → Dropout 이후
    두 개의 헤드(head_mean, head_tail). head_tail은 softplus로 양수 보장.
    MC 추론 시 불확실성 추정을 위해 Dropout을 활성 상태로 유지합니다.
    손실: SmoothL1(mean, y_mean) + quantile_w * QuantileLoss(p95, y_p95; τ=0.95)

    매 에폭 전체셋 MAE/R2를 계산해 v27.3와 동일한 로그 포맷을 출력하며,
    학습 종료 시 [TRAIN] Wall time / 최종 train_MAE / R2 / sanity corr
    3줄 요약을 추가로 출력합니다.

    --------------------------------------------------------------------------
    Mean + tail head (p95 = mean + softplus(tail)).
    With CUDA, training tensors (X, y_mean, y_p95) are moved to GPU once and
    mini-batches are formed via GPU indexing each epoch for maximum speed.
    If CPU is required, a DataLoader path (pinned memory, multi-worker) is used.

    Architecture: Linear→SiLU→Dropout→Linear→SiLU→Dropout, then two heads
    (head_mean, head_tail with softplus). Dropout remains active during MC
    inference to estimate uncertainty. Loss = SmoothL1(mean, y_mean) +
    quantile_w * QuantileLoss(p95, y_p95; τ=0.95).

    Per epoch, MAE/R2 on the full set are logged in the same format as v27.3,
    and a final 3-line summary ([TRAIN] Wall time / train_MAE / R2 / sanity corr)
    is printed at the end.
    """

    def __init__(self, in_dim, hidden=128, p=0.1, lr=3e-4, weight_decay=1e-4,
                 quantile_w=0.3, device=None, force_cpu_loader: bool = False):
        super().__init__()
        self.fc1=nn.Linear(in_dim, hidden); self.fc2=nn.Linear(hidden, hidden)
        self.do=nn.Dropout(p); self.act=nn.SiLU()
        self.head_mean=nn.Linear(hidden,1); self.head_tail=nn.Linear(hidden,1)
        self.lr=lr; self.wd=weight_decay; self.quantile_w=quantile_w
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.force_cpu_loader = force_cpu_loader

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def forward(self, x, mc=False):
        if mc:  # MC-dropout
            self.train(True)
        h=self.act(self.fc1(x)); h=self.do(h)
        h=self.act(self.fc2(h)); h=self.do(h)
        mean=self.head_mean(h).squeeze(-1)
        tail=F.softplus(self.head_tail(h)).squeeze(-1)
        return mean, tail

    @staticmethod
    def _qloss(y_pred, y_true, tau=0.95):
        e = y_true - y_pred
        return torch.mean(torch.maximum(tau*e, (tau-1)*e))

    def _epoch_metrics(self, Xt_full, ym_full):
        self.eval()
        with torch.no_grad():
            pm_full, _ = self(Xt_full, mc=False)
            mae = torch.mean(torch.abs(pm_full - ym_full)).item()
            ss_res = torch.sum((pm_full - ym_full)**2).item()
            ss_tot = torch.sum((ym_full - torch.mean(ym_full))**2).item() + 1e-9
            r2 = 1.0 - ss_res/ss_tot
        return mae, r2, pm_full

    def fit(self, X, y_mean, y_p95, epochs=60, batch=4096):
        """
        v27.3 재현: 매 에폭마다 전체셋 MAE/R2 계산(총 150회), 요약 로그 3줄 포함.
        """
        use_gpu_dataset = (self.device == 'cuda') and (not self.force_cpu_loader)
        self.to(self.device)

        if use_gpu_dataset:
            # === 빠른 경로: 데이터 한 번만 GPU로 전송, 에폭마다 GPU 슬라이싱 ===
            Xg  = torch.as_tensor(X,      dtype=torch.float32, device=self.device).contiguous()
            ymg = torch.as_tensor(y_mean, dtype=torch.float32, device=self.device).contiguous()
            ypg = torch.as_tensor(y_p95,  dtype=torch.float32, device=self.device).contiguous()
            N = Xg.size(0)
            Xt_full, ym_full = Xg, ymg

            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
            log("=== Train compute regressor (ms targets; mean + p95 quantile) ===")
            t0 = time.time()
            for ep in range(1, epochs+1):
                self.train(); loss_sum=0.0; n=0
                # shuffle on GPU
                perm = torch.randperm(N, device=self.device)
                for s in range(0, N, batch):
                    idx = perm[s:s+batch]
                    xb   = Xg.index_select(0, idx)
                    ym_b = ymg.index_select(0, idx)
                    yp_b = ypg.index_select(0, idx)

                    opt.zero_grad()
                    pm, tail = self(xb, mc=False); pp95 = pm + tail
                    loss = F.smooth_l1_loss(pm, ym_b) + self.quantile_w*self._qloss(pp95, yp_b, 0.95)
                    loss.backward(); opt.step()

                    loss_sum += loss.item()*xb.size(0); n += xb.size(0)

                mae, r2, _ = self._epoch_metrics(Xt_full, ym_full)
                avg_loss = loss_sum / max(1, n)
                log(f"epoch {ep:4d}/{epochs} | loss {avg_loss:.4f} | MAE {mae:.3f} ms | R2 {r2:.4f}")

            wall = time.time() - t0
            log(f"[TRAIN] Wall time: {wall:.2f} s")
            mae, r2, pm_full = self._epoch_metrics(Xt_full, ym_full)
            log(f"epoch {epochs:4d}/{epochs} | train_MAE {mae:.3f} ms | R2 {r2:.4f}")
            y_np  = ym_full.detach().cpu().numpy()
            pm_np = pm_full.detach().cpu().numpy()
            corr = np.corrcoef(y_np, pm_np)[0,1]
            log(f"[sanity] corr(pred_compute_mean, compute_proxy) TRAIN = {corr:.3f}")
            return self

        else:
            # === CPU DataLoader 경로(느리지만 메모리 제한 시 선택) ===
            X_cpu  = torch.as_tensor(X,      dtype=torch.float32, device='cpu').contiguous()
            ym_cpu = torch.as_tensor(y_mean, dtype=torch.float32, device='cpu').contiguous()
            yp_cpu = torch.as_tensor(y_p95,  dtype=torch.float32, device='cpu').contiguous()

            pin = (self.device == 'cuda')
            cpu_cnt = os.cpu_count() or 4
            nw = max(1, min(8, cpu_cnt // 2))
            dl = DataLoader(
                TensorDataset(X_cpu, ym_cpu, yp_cpu),
                batch_size=batch, shuffle=True,
                pin_memory=pin, num_workers=nw,
                persistent_workers=True, prefetch_factor=4
            )

            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
            Xt_full = X_cpu.to(self.device, non_blocking=pin)
            ym_full = ym_cpu.to(self.device, non_blocking=pin)

            log("=== Train compute regressor (ms targets; mean + p95 quantile) ===")
            t0 = time.time()
            for ep in range(1, epochs+1):
                self.train(); loss_sum=0.0; n=0
                for xb_c, ym_c, yp_c in dl:
                    xb   = xb_c.to(self.device, non_blocking=pin)
                    ym_b = ym_c.to(self.device, non_blocking=pin)
                    yp_b = yp_c.to(self.device, non_blocking=pin)

                    opt.zero_grad()
                    pm, tail = self(xb, mc=False); pp95 = pm + tail
                    loss = F.smooth_l1_loss(pm, ym_b) + self.quantile_w*self._qloss(pp95, yp_b, 0.95)
                    loss.backward(); opt.step()

                    loss_sum += loss.item()*len(xb); n += len(xb)

                mae, r2, _ = self._epoch_metrics(Xt_full, ym_full)
                avg_loss = loss_sum / max(1, n)
                log(f"epoch {ep:4d}/{epochs} | loss {avg_loss:.4f} | MAE {mae:.3f} ms | R2 {r2:.4f}")

            wall = time.time() - t0
            log(f"[TRAIN] Wall time: {wall:.2f} s")
            mae, r2, pm_full = self._epoch_metrics(Xt_full, ym_full)
            log(f"epoch {epochs:4d}/{epochs} | train_MAE {mae:.3f} ms | R2 {r2:.4f}")
            y_np  = ym_full.detach().cpu().numpy()
            pm_np = pm_full.detach().cpu().numpy()
            corr = np.corrcoef(y_np, pm_np)[0,1]
            log(f"[sanity] corr(pred_compute_mean, compute_proxy) TRAIN = {corr:.3f}")
            return self

    def predict(self, X, T=16):
        x = torch.tensor(X, dtype=torch.float32, device=self.device); self.eval()
        if T <= 1:
            with torch.no_grad():
                pm, tail = self(x, mc=False)
                return pm.cpu().numpy(), (pm+tail).cpu().numpy(), np.zeros(len(X))
        ms, ps = [], []
        for _ in range(T):
            with torch.no_grad():
                pm, tail = self(x, mc=True)
                ms.append(pm.cpu().numpy()); ps.append((pm+tail).cpu().numpy())
        ms = np.stack(ms); ps = np.stack(ps)
        return ms.mean(0), ps.mean(0), ms.std(0)
