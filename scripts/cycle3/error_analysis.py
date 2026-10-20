"""Word-level error analysis of a model (ensemble over seeds) on saved dev/test probabilities."""
import glob, os, sys, json
import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/cefr-lexicon-kaz/results/predictions"))
L = ["A1", "A2", "B1", "B2", "C1"]
M = {l: i for i, l in enumerate(L)}
tag = sys.argv[1]
suffix = sys.argv[2] if len(sys.argv) > 2 else "_dev.csv"
Ds = []
for f in sorted(glob.glob(f"*_{tag}_s*{suffix}")):
    Ds.append(pd.read_csv(f, encoding="utf-8-sig"))
print("files:", len(Ds))
ref = Ds[0].copy()
P = np.mean([d[[f"prob_{l}" for l in L]].to_numpy() for d in Ds], 0)
y = ref["gold_cefr"].map(M).to_numpy()
pred = P.argmax(1)
ref["pred_ens"] = [L[i] for i in pred]
ref["dist"] = np.abs(pred - y)
ref["signed"] = pred - y
ref["p_gold"] = P[np.arange(len(y)), y]
ref["conf"] = P.max(1)
# agreement between seeds (share of seeds with the same predicted label as the ensemble)
seed_preds = np.array([d[[f"prob_{l}" for l in L]].to_numpy().argmax(1) for d in Ds])
ref["agree"] = (seed_preds == pred[None, :]).mean(0)
print("ensemble accuracy %.4f within1 %.4f" % ((pred == y).mean(), (ref["dist"] <= 1).mean()))
print("\nerror distance:", ref["dist"].value_counts(normalize=True).sort_index().round(3).to_dict())
print("\nby POS (n, acc, within1, mean|err|):")
g = ref.groupby("pos").agg(n=("dist", "size"), acc=("dist", lambda s: (s == 0).mean()), w1=("dist", lambda s: (s <= 1).mean()), mae=("dist", "mean"))
print(g.sort_values("n", ascending=False).round(3).to_string())
print("\nby context availability:")
print(ref.groupby("context_available").agg(n=("dist", "size"), acc=("dist", lambda s: (s == 0).mean()), w1=("dist", lambda s: (s <= 1).mean())).round(3).to_string())
ref["len"] = ref["lemma"].str.len()
ref["lenbin"] = pd.cut(ref["len"], [0, 4, 6, 8, 11, 40])
print("\nby lemma length:")
print(ref.groupby("lenbin", observed=True).agg(n=("dist", "size"), acc=("dist", lambda s: (s == 0).mean()), w1=("dist", lambda s: (s <= 1).mean())).round(3).to_string())
far = ref[ref["dist"] >= 3].sort_values("conf", ascending=False)
print("\nconfident errors of 3+ levels: n =", len(far))
print(far[["lemma", "pos", "cefr", "pred_ens", "conf", "agree"]].head(14).round(2).to_string(index=False))
over = ref[ref["signed"] >= 2].sort_values("p_gold").head(8)
under = ref[ref["signed"] <= -2].sort_values("p_gold").head(8)
print("\noverestimated by 2+ levels (n=%d):" % (ref["signed"] >= 2).sum()); print(over[["lemma", "pos", "cefr", "pred_ens", "conf", "agree"]].round(2).to_string(index=False))
print("\nunderestimated by 2+ levels (n=%d):" % (ref["signed"] <= -2).sum()); print(under[["lemma", "pos", "cefr", "pred_ens", "conf", "agree"]].round(2).to_string(index=False))
# stable errors: same wrong label in all seeds
stable = ref[(ref["dist"] >= 2) & (ref["agree"] == 1.0)]
print("\nerrors of 2+ levels identical in all seeds: n =", len(stable), "of", int((ref["dist"] >= 2).sum()))
ref.to_csv(os.path.expanduser(f"~/errwords_{tag}.csv"), index=False, encoding="utf-8-sig")
