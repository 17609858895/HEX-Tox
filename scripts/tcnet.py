"""Toxicity-curve network (TC-Net): embedding trunk + structured duration heads, censored Gaussian likelihood.

Heads (all predict mu = log10 LC50 at duration t, hours):
  free  : mu = MLP([h, log t])                                   (unconstrained ablation)
  haber : mu = a(h) - n(h) * (log t - log 48)                    (Haber / log-linear toxicity curve)
  curve : mu = log10 LCinf(h) + log10(1 + (tau(h)/t)^n(h))         (saturating toxicity curve, incipient level LCinf)
Censoring: exact -> Gaussian density; right ('>') -> survival; left ('<') -> CDF.
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn

LOG48 = math.log10(48.0)


class Encoder:
    def __init__(self, cats, nums):
        self.cats, self.nums = cats, nums

    def fit(self, df):
        self.maps = {c: {v: i + 1 for i, v in enumerate(df[c].astype(str).unique())} for c in self.cats}
        self.mu = df[self.nums].mean(); self.sd = df[self.nums].std().replace(0, 1)
        return self

    def transform(self, df):
        C = np.stack([df[c].astype(str).map(self.maps[c]).fillna(0).astype(int).values for c in self.cats], 1)
        Z = ((df[self.nums] - self.mu) / self.sd)
        M = Z.isna().astype(float).values
        Z = Z.fillna(0).values
        return torch.tensor(C, dtype=torch.long), torch.tensor(np.hstack([Z, M]), dtype=torch.float32)


class TCNet(nn.Module):
    def __init__(self, card, n_num, head="curve", emb=16, hid=256, drop=0.15, n_sig=1, anchor=False):
        super().__init__()
        self.head, self.anchor = head, anchor
        self.embs = nn.ModuleList([nn.Embedding(k + 1, min(emb, max(2, int(1.6 * k ** 0.56)))) for k in card])
        d = sum(e.embedding_dim for e in self.embs) + n_num
        self.trunk = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(hid, hid), nn.GELU(), nn.Dropout(drop))
        if head == "free":
            self.out = nn.Sequential(nn.Linear(hid + 1, hid // 2), nn.GELU(), nn.Linear(hid // 2, 1))
        elif head == "haber":
            self.out = nn.Linear(hid, 2)
        else:
            self.out = nn.Linear(hid, 3)
        self.log_sigma = nn.Parameter(torch.zeros(n_sig) - 0.3)
        if anchor:  # within-study (Mundlak) temperature slope: beta(h) * (T - mean_T of study series)
            self.beta = nn.Linear(hid, 1)
            nn.init.zeros_(self.beta.weight); nn.init.constant_(self.beta.bias, 0.0)

    def params(self, C, Z):
        e = torch.cat([emb(C[:, i]) for i, emb in enumerate(self.embs)] + [Z], 1)
        return self.trunk(e)

    def forward(self, C, Z, logt, dT=None, return_parts=False):
        h = self.params(C, Z)
        if self.head == "free":
            mu = self.out(torch.cat([h, (logt - LOG48).unsqueeze(1)], 1)).squeeze(1)
            parts = {}
        elif self.head == "haber":
            o = self.out(h)
            a, n = o[:, 0], nn.functional.softplus(o[:, 1])
            mu = a - n * (logt - LOG48)
            parts = dict(a=a, n=n)
        else:
            o = self.out(h)
            linf = o[:, 0]
            logtau = o[:, 1] + 1.5       # log10 tau (h), init ~ 30 h
            n = nn.functional.softplus(o[:, 2]) + 0.05
            # log10(1 + (tau/t)^n) computed stably via softplus in natural log
            z = n * (logtau - logt) * math.log(10.0)
            mu = linf + nn.functional.softplus(z) / math.log(10.0)
            parts = dict(linf=linf, logtau=logtau, n=n)
        if self.anchor:
            b = self.beta(h).squeeze(1) * 0.01  # log10 units per degC
            mu = mu + b * dT
            parts["beta"] = b
        return (mu, parts) if return_parts else mu


def censored_nll(mu, y, cens, log_sigma):
    """cens: 0 exact, 1 right (true > y), 2 left (true < y)."""
    sig = torch.exp(log_sigma)
    z = (y - mu) / sig
    nd = torch.distributions.Normal(0.0, 1.0)
    ll_exact = nd.log_prob(z) - torch.log(sig)
    ll_right = torch.log(torch.clamp(1 - nd.cdf(z), min=1e-8))
    ll_left = torch.log(torch.clamp(nd.cdf(z), min=1e-8))
    ll = torch.where(cens == 0, ll_exact, torch.where(cens == 1, ll_right, ll_left))
    return -ll.mean()


def fit_tcnet(tr, cats, nums, head="curve", epochs=40, bs=2048, lr=2e-3, wd=1e-5, seed=0, use_cens=True,
              sig_group=None, verbose=False, anchor_col=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if not use_cens:
        tr = tr[tr.cens == "exact"]
    enc = Encoder(cats, nums).fit(tr)
    C, Z = enc.transform(tr)
    t = torch.tensor(tr.log_dur.values, dtype=torch.float32)
    y = torch.tensor(tr.log10_mgL.values, dtype=torch.float32)
    cz = torch.tensor(tr.cens.map({"exact": 0, "right": 1, "left": 2}).values, dtype=torch.long)
    card = [len(enc.maps[c]) for c in cats]
    dT = torch.tensor(tr[anchor_col].fillna(0).values if anchor_col else np.zeros(len(tr)), dtype=torch.float32)
    model = TCNet(card, Z.shape[1], head=head, anchor=anchor_col is not None)
    model.anchor_col = anchor_col
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(len(y) / bs))
    n = len(y)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n); tot = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            mu = model(C[idx], Z[idx], t[idx], dT[idx])
            loss = censored_nll(mu, y[idx], cz[idx], model.log_sigma[0])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
        if verbose and ep % 10 == 0:
            print(ep, tot / n, float(torch.exp(model.log_sigma[0])))
    model.eval()
    return model, enc


@torch.no_grad()
def predict_tcnet(model, enc, df, logt=None, parts=False, dT=None):
    C, Z = enc.transform(df)
    t = torch.tensor(df.log_dur.values if logt is None else np.broadcast_to(logt, len(df)).copy(), dtype=torch.float32)
    ac = getattr(model, "anchor_col", None)
    if dT is None:
        dT = df[ac].fillna(0).values if ac else np.zeros(len(df))
    dT = torch.tensor(np.broadcast_to(dT, len(df)).copy(), dtype=torch.float32)
    mu, p = model(C, Z, t, dT, return_parts=True)
    if parts:
        return mu.numpy(), {k: v.numpy() for k, v in p.items()}
    return mu.numpy()
