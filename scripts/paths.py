"""Project paths — ALL derived from this file's location, so the whole project is RELOCATABLE.
Move the project root anywhere (keeping data/ inside) and every script still works."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # parent of scripts/
DATA = os.path.join(ROOT, "data")
AMOS = os.path.join(DATA, "amos_mri")
MRISEG_REL = os.path.join(DATA, "mrisegmentator", "MRISegmenter_T1only_public_20Jun2025", "Release")
OUT = os.path.join(ROOT, "outputs")
PLOTS = os.path.join(OUT, "plots")
RESULTS = os.path.join(OUT, "results")
LOGS = os.path.join(OUT, "logs")
for _d in (PLOTS, RESULTS, LOGS):
    os.makedirs(_d, exist_ok=True)
