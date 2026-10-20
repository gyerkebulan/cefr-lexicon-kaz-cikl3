import glob, json, os, sys
import numpy as np
from scipy import stats
os.chdir(os.path.expanduser("~/cefr-lexicon-kaz"))
tags = sys.argv[1:]
L = ["A1", "A2", "B1", "B2", "C1"]
R = {}
for f in sorted(glob.glob("results/metrics/*_dev.json")):
    n = os.path.basename(f)
    for t in tags:
        key = f"_{t}_s"
        if key in n:
            seed = int(n.split(key)[1].split("_")[0].split(".")[0])
            R.setdefault(t, {})[seed] = json.load(open(f, encoding="utf-8"))
            break


def f1s(d):
    cm = np.array(d["confusion_matrix"], float)
    tp = np.diag(cm); rec = tp / np.maximum(cm.sum(1), 1); pre = tp / np.maximum(cm.sum(0), 1)
    return 2 * pre * rec / np.maximum(pre + rec, 1e-12)


print(f"{'tag':16}{'n':>3}{'Acc':>8}{'sd':>7}{'MF1':>8}{'sd':>7}{'MAE':>7}{'QWK':>7}{'ECE':>7}{'W1':>7}{'Brier':>7}{'NLL':>7}  F1 A1..C1")
for t in tags:
    if t not in R:
        print(f"{t:16} none"); continue
    ds = [R[t][s] for s in sorted(R[t])]
    g = lambda k: float(np.mean([x[k] for x in ds]))
    sd = lambda k: float(np.std([x[k] for x in ds], ddof=1)) if len(ds) > 1 else 0.0
    nll = g("negative_log_likelihood") if "negative_log_likelihood" in ds[0] else float("nan")
    f1 = np.mean([f1s(x) for x in ds], 0)
    print(f"{t:16}{len(ds):3}{g('accuracy'):8.4f}{sd('accuracy'):7.4f}{g('macro_f1'):8.4f}{sd('macro_f1'):7.4f}{g('ordinal_mae'):7.3f}"
          f"{g('quadratic_weighted_kappa'):7.3f}{g('expected_calibration_error'):7.3f}{1 - g('far_error_rate'):7.3f}{g('brier_score'):7.3f}{nll:7.3f}  "
          + " ".join(f"{v:.3f}" for v in f1))
base = tags[0]
if base in R:
    print(f"\nWelch vs {base}:")
    for t in tags[1:]:
        if t not in R:
            continue
        out = []
        for k, nm in (("accuracy", "Acc"), ("macro_f1", "MF1"), ("expected_calibration_error", "ECE"), ("far_error_rate", "W1")):
            x = np.array([R[t][s][k] for s in sorted(R[t])]); y = np.array([R[base][s][k] for s in sorted(R[base])])
            sgn = -1 if k == "far_error_rate" else 1
            out.append(f"{nm} d={sgn * (x.mean() - y.mean()):+.4f} p={stats.ttest_ind(x, y, equal_var=False).pvalue:.3f}")
        print(f"  {t:16} " + " | ".join(out))
