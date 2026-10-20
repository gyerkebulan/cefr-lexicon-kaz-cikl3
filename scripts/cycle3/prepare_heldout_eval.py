"""Directory in which the sealed cycle 3 test part takes the place of dev, so that frozen checkpoints
can be evaluated once with the ordinary evaluation path. Train part stays identical (feature statistics)."""
import os, shutil
os.chdir(os.path.expanduser("~/cefr-lexicon-kaz"))
SRC = "dataset/research/ar-cycle3-clean3563-20261020"
OUT = "dataset/research/ar-cycle3-testeval"
os.makedirs(OUT, exist_ok=True)
for fn in os.listdir(SRC):
    shutil.copy(f"{SRC}/{fn}", f"{OUT}/{fn}")
shutil.copy(f"{SRC}/test.csv", f"{OUT}/dev.csv")
shutil.copy(f"{SRC}/test_contexts.json", f"{OUT}/dev_contexts.json")
print("ready", OUT, sorted(os.listdir(OUT)))
