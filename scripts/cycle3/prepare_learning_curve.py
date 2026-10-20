"""Nested group-wise subsamples of the cycle 3 train split (learning curve)."""
import os, random, shutil
import pandas as pd

os.chdir(os.path.expanduser("~/cefr-lexicon-kaz"))
SRC = "dataset/research/ar-cycle3-clean3563-20261020"
tr = pd.read_csv(f"{SRC}/train.csv", encoding="utf-8-sig")
groups = sorted(tr["group_id"].unique())
random.Random(0).shuffle(groups)
sizes = tr.groupby("group_id").size().to_dict()
total = len(tr)
for frac in (25, 50, 75):
    target = total * frac / 100
    chosen, n = set(), 0
    for g in groups:
        if n >= target:
            break
        chosen.add(g); n += sizes[g]
    out = f"dataset/research/ar-cycle3-lc{frac}"
    os.makedirs(out, exist_ok=True)
    for fn in os.listdir(SRC):
        if fn != "train.csv" and not fn.startswith("train_contexts"):
            shutil.copy(f"{SRC}/{fn}", f"{out}/{fn}")
    sub = tr[tr["group_id"].isin(chosen)]
    sub.to_csv(f"{out}/train.csv", index=False, encoding="utf-8-sig")
    print(frac, len(sub), "rows", sub["cefr"].value_counts().sort_index().to_dict())
