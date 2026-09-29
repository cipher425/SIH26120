"""PyTorch models: M2 LSTM production forecaster and M3 1-D CNN dyno-card classifier."""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(max(1, torch.get_num_threads()))

M2_FEATS = ["oil_bbl", "water_bbl", "T_wellhead", "T_pump", "spm", "prod_day"]
WIN, HOR = 21, 14


class LSTMForecaster(nn.Module):
    def __init__(self, n_in=len(M2_FEATS), hidden=96, horizon=HOR):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def m2_transform(block: np.ndarray) -> np.ndarray:
    """block columns follow M2_FEATS. Returns normalised features."""
    x = np.empty_like(block, dtype=np.float32)
    x[:, 0] = np.log1p(block[:, 0])
    x[:, 1] = np.log1p(block[:, 1])
    x[:, 2] = block[:, 2] / 100
    x[:, 3] = block[:, 3] / 100
    x[:, 4] = block[:, 4] / 10
    x[:, 5] = block[:, 5] / 300
    return x


def decline_baseline(X):
    """Exponential (Arps b=0) decline fitted to the window's log-oil, returned as
    log-change relative to the last observed day, for each horizon step."""
    t = np.arange(WIN, dtype=float)
    tf = np.arange(WIN, WIN + HOR, dtype=float)
    y = X[:, :, 0]
    tm = t.mean()
    b = ((t - tm) * (y - y.mean(1, keepdims=True))).sum(1) / ((t - tm) ** 2).sum()
    a = y.mean(1) - b * tm
    fut = a[:, None] + b[:, None] * tf[None, :]
    return (fut - y[:, -1:]).astype(np.float32)


def forecast(model, X):
    """Hybrid forecast: decline-curve physics + LSTM-learned correction (log space)."""
    with torch.no_grad():
        corr = model(torch.as_tensor(X)).numpy()
    return decline_baseline(X) + corr


class CardCNN(nn.Module):
    def __init__(self, n_cls=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3, padding_mode="circular"), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2, padding_mode="circular"), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 3, padding=1, padding_mode="circular"), nn.BatchNorm1d(96), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, n_cls))

    def forward(self, x):
        return self.net(x)


def train_loop(model, X, y, epochs, lr=2e-3, bs=256, loss_fn=None, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X = torch.as_tensor(X)
    y = torch.as_tensor(y)
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X[b]), y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        print(f"   epoch {ep + 1}/{epochs} loss {tot / n:.4f}")
    model.eval()
    return model
