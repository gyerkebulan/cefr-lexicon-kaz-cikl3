"""Post-hoc decision/calibration approaches on saved dev probabilities, cross-fitted over two group halves."""
import glob, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score

os.chdir(os.path.expanduser("~/cefr-lexicon-kaz/results/predictions"))
L = ["A1", "A2", "B1", "B2", "C1"]
M = {l: i for i, l in enumerate(L)}
tags = sys.argv[1:]


def load(tag):
    out = {}
    for f in sorted(glob.glob("*_dev.csv")):
        if f"_{tag}_s" in f:
            seed = int(f.split(f"_{tag}_s")[1].split("_")[0].split(".")[0])
            d = pd.read_csv(f, encoding="utf-8-sig")
            out[seed] = d
    return out


def metrics(P, y, pred):
    acc = float((pred == y).mean())
    mf1 = float(f1_score(y, pred, average="macro", labels=range(5), zero_division=0))
    mae = float(np.abs(pred - y).mean())
    qwk = float(cohen_kappa_score(y, pred, weights="quadratic"))
    w1 = float((np.abs(pred - y) <= 1).mean())
    conf = P.max(1)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())
    return dict(acc=acc, mf1=mf1, mae=mae, qwk=qwk, w1=w1, ece=float(ece), nll=nll)


def temp_scale(P, T):
    Q = np.power(np.clip(P, 1e-12, 1), 1.0 / T)
    return Q / Q.sum(1, keepdims=True)


def fit_T(P, y):
    grid = np.arange(0.5, 4.01, 0.1)
    return grid[int(np.argmin([-np.log(np.clip(temp_scale(P, t)[np.arange(len(y)), y], 1e-12, 1)).mean() for t in grid]))]


def median_decode(P):
    return (np.cumsum(P, 1) >= 0.5).argmax(1)


def fit_bias(P, y):
    b = np.zeros(5)
    lp = np.log(np.clip(P, 1e-12, 1))
    best = f1_score(y, (lp + b).argmax(1), average="macro", labels=range(5), zero_division=0)
    for _ in range(3):
        for c in range(5):
            for v in np.arange(-1.0, 1.01, 0.1):
                bb = b.copy(); bb[c] = v
                s = f1_score(y, (lp + bb).argmax(1), average="macro", labels=range(5), zero_division=0)
                if s > best + 1e-9:
                    best, b = s, bb
    return b


def evaluate_variants(P, y, fold):
    """P: (n,5) probabilities, y labels, fold: 0/1 per row (group halves)."""
    res = {}
    res["argmax"] = metrics(P, y, P.argmax(1))
    res["median"] = metrics(P, y, median_decode(P))
    Pt = P.copy(); bias = np.zeros_like(P)
    for k in (0, 1):
        tr, te = fold != k, fold == k
        Pt[te] = temp_scale(P[te], fit_T(P[tr], y[tr]))
        bias[te] = fit_bias(P[tr], y[tr])
    res["temp"] = metrics(Pt, y, Pt.argmax(1))
    res["temp+median"] = metrics(Pt, y, median_decode(Pt))
    pb = (np.log(np.clip(P, 1e-12, 1)) + bias).argmax(1)
    res["bias"] = metrics(P, y, pb)
    return res


for tag in tags:
    D = load(tag)
    if not D:
        print(tag, "none"); continue
    seeds = sorted(D)
    ref = D[seeds[0]]
    y = ref["gold_cefr"].map(M).to_numpy()
    fold = (ref["group_id"].to_numpy() % 2).astype(int)
    per = {}
    for s in seeds:
        P = D[s][[f"prob_{l}" for l in L]].to_numpy()
        for k, v in evaluate_variants(P, y, fold).items():
            per.setdefault(k, []).append(v)
    Pens = np.mean([D[s][[f"prob_{l}" for l in L]].to_numpy() for s in seeds], 0)
    ens = evaluate_variants(Pens, y, fold)
    print(f"\n== {tag} (n seeds = {len(seeds)})")
    print(f"{'variant':14}{'Acc':>8}{'MF1':>8}{'MAE':>7}{'QWK':>7}{'ECE':>7}{'W1':>7}{'NLL':>7}")
    for k, lst in per.items():
        m = {kk: float(np.mean([x[kk] for x in lst])) for kk in lst[0]}
        print(f"{k:14}{m['acc']:8.4f}{m['mf1']:8.4f}{m['mae']:7.3f}{m['qwk']:7.3f}{m['ece']:7.3f}{m['w1']:7.3f}{m['nll']:7.3f}")
    for k in ("argmax", "median", "temp", "bias"):
        m = ens[k]
        print(f"{'ens:' + k:14}{m['acc']:8.4f}{m['mf1']:8.4f}{m['mae']:7.3f}{m['qwk']:7.3f}{m['ece']:7.3f}{m['w1']:7.3f}{m['nll']:7.3f}")
