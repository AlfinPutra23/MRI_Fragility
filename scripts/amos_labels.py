"""AMOS22-MRI label map (verified 2026-06-30 from the dataset jsonl + label-value audit).
Same TAIL organs as MRISegmentator (gallbladder, esophagus, pancreas, adrenals, duodenum) -> clean
generalization test of the fragility ordering."""

LABELS = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "esophagus",
          6: "liver", 7: "stomach", 8: "aorta", 9: "inferior_vena_cava", 10: "pancreas",
          11: "adrenal_gland_right", 12: "adrenal_gland_left", 13: "duodenum",
          14: "urinary_bladder", 15: "prostate_uterus"}

# the 11 abdominal organs we benchmark (AMOS lacks small_bowel/colon vs MRISegmentator)
ABDO = {1: "spleen", 2: "kidney_R", 3: "kidney_L", 4: "gallbladder", 5: "esophagus", 6: "liver",
        7: "stomach", 10: "pancreas", 11: "adrenal_R", 12: "adrenal_L", 13: "duodenum"}
# small/hard tail organs (same set as MRISegmentator)
TAIL = {4, 5, 10, 11, 12, 13}   # gallbladder, esophagus, pancreas, adrenal_R/L, duodenum

SHORT = {**ABDO, 8: "aorta", 9: "IVC", 14: "bladder"}


def dataset_json_labels():
    d = {"background": 0}
    for k, v in LABELS.items():
        d[v] = k
    return d
