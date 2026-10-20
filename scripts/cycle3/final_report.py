"""Final held-out results: single-network means, ensembles, temperature fitted on dev only, bootstrap CIs and paired tests."""
import glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

os.chdir(os.path.expanduser("~/cefr-lexicon-kaz/results/predictions"))
L = ["A1", "A2", "B1", "B2", "C1"]
M = {l: i for i, l in enumerate(L)}
MODELS = {  # name: (dev tag, test tag)
    "V2*": ("c3v2star", "c3tV2"), "V3": ("c3v3", "c3tV3"), "V4": ("c3v4", "c3tV4"), "V5": ("c3v5", "c3tV5"), "V6": ("c4p4v5", "c3tV6"),
}
rng = np.random.default_rng(0)


def load(tag):
    out = {}
    for f in sorted(glob.glob("*_dev.csv")):
        if f"_{tag}_s" in f:
            seed = int(f.split(f"_{tag}_s")[1].split("_")[0].split(".")[0])
            out[seed] = pd.read_csv(f, encoding="utf-8-sig")
    return out


def probs(d):
    return d[[f"prob_{l}" for l in L]].to_numpy()


def temp(P, T):
    Q = np.power(np.clip(P, 1e-12, 1), 1.0 / T)
    return Q / Q.sum(1, keepdims=True)


def fit_T(P, y):
    grid = np.arange(0.5, 4.01, 0.05)
    return float(grid[int(np.argmin([-np.log(np.clip(temp(P, t)[np.arange(len(y)), y], 1e-12, 1)).mean() for t in grid]))])


def met(P, y):
    pred = P.argmax(1)
    conf = P.max(1)
    bins = np.linspace(0, 1, 16); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return dict(acc=(pred == y).mean(), mf1=f1_score(y, pred, average="macro", labels=range(5), zero_division=0),
                mae=np.abs(pred - y).mean(), qwk=cohen_kappa_score(y, pred, weights="quadratic"),
                w1=(np.abs(pred - y) <= 1).mean(), ece=ece, nll=-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


res = {}
for name, (dt, tt) in MODELS.items():
    Dd, Dt = load(dt), load(tt)
    if not Dt:
        continue
    ytest = Dt[sorted(Dt)[0]]["gold_cefr"].map(M).to_numpy()
    ydev = Dd[sorted(Dd)[0]]["gold_cefr"].map(M).to_numpy()
    singles = [met(probs(Dt[s]), ytest) for s in sorted(Dt)]
    Pdev = np.mean([probs(Dd[s]) for s in sorted(Dd)], 0)
    Ptest = np.mean([probs(Dt[s]) for s in sorted(Dt)], 0)
    T = fit_T(Pdev, ydev)
    res[name] = dict(P=Ptest, T=T, y=ytest, singles=singles, n=len(Dt))
    sm = {k: np.mean([x[k] for x in singles]) for k in singles[0]}
    ssd = {k: np.std([x[k] for x in singles], ddof=1) if len(singles) > 1 else 0 for k in singles[0]}
    e = met(Ptest, ytest); et = met(temp(Ptest, T), ytest)
    print(f"\n== {name}: test words {len(ytest)}, networks {len(Dt)}, T(dev)={T:.2f}")
    print(f"single mean  acc {sm['acc']:.4f}±{ssd['acc']:.4f} mf1 {sm['mf1']:.4f}±{ssd['mf1']:.4f} mae {sm['mae']:.3f} qwk {sm['qwk']:.3f} ece {sm['ece']:.3f} w1 {sm['w1']:.3f} nll {sm['nll']:.3f}")
    print(f"ensemble     acc {e['acc']:.4f} mf1 {e['mf1']:.4f} mae {e['mae']:.3f} qwk {e['qwk']:.3f} ece {e['ece']:.3f} w1 {e['w1']:.3f} nll {e['nll']:.3f}")
    print(f"ens + T(dev) acc {et['acc']:.4f} mf1 {et['mf1']:.4f} mae {et['mae']:.3f} qwk {et['qwk']:.3f} ece {et['ece']:.3f} w1 {et['w1']:.3f} nll {et['nll']:.3f}")
    pred = Ptest.argmax(1)
    prf = f1_score(ytest, pred, average=None, labels=range(5), zero_division=0)
    print("ensemble F1 by level:", " ".join(f"{l} {v:.3f}" for l, v in zip(L, prf)))
    # bootstrap CI by words
    B = 1000; idx = rng.integers(0, len(ytest), (B, len(ytest)))
    accs = [(pred[i] == ytest[i]).mean() for i in idx]
    w1s = [(np.abs(pred[i] - ytest[i]) <= 1).mean() for i in idx]
    print("ensemble 95%% CI acc [%.3f, %.3f] within1 [%.3f, %.3f]" % (*np.percentile(accs, [2.5, 97.5]), *np.percentile(w1s, [2.5, 97.5])))
    if name == "V6":
        cm = confusion_matrix(ytest, pred, labels=range(5))
        print("confusion (rows true):"); print(cm)
        print("error distance:", {k: round(float((np.abs(pred - ytest) == k).mean()), 3) for k in range(5)})

if "V6" in res:
    y = res["V6"]["y"]
    print("\nPaired bootstrap of ensembles, V6 minus other (95% CI):")
    p6 = res["V6"]["P"].argmax(1)
    B = 2000; idx = rng.integers(0, len(y), (B, len(y)))
    for other in ("V2*", "V3", "V4", "V5"):
        if other not in res:
            continue
        po = res[other]["P"].argmax(1)
        for nm, fn in (("acc", lambda p, i: (p[i] == y[i]).mean()), ("within1", lambda p, i: (np.abs(p[i] - y[i]) <= 1).mean()),
                       ("mf1", lambda p, i: f1_score(y[i], p[i], average="macro", labels=range(5), zero_division=0))):
            d = np.array([fn(p6, i) - fn(po, i) for i in idx[:1000 if nm == "mf1" else B]])
            print(f"  V6-{other} {nm}: {d.mean():+.4f} CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
