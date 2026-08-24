"""Authoritative MRISegmentator-Abdomen label map (repo README / ITK label file, verified 2026-06-28).
The arXiv paper's PROSE order is NOT the index order -> always use this."""

_RIBS_L = {33 + i: f"rib_left_{r}" for i, r in enumerate(range(4, 13), start=1)}     # 34..42
_RIBS_R = {42 + i: f"rib_right_{r}" for i, r in enumerate(range(4, 13), start=1)}    # 43..51
_VERT = {52 + i: n for i, n in enumerate(
    ["vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
     "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7"])}

LABELS = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
    6: "esophagus", 7: "stomach", 8: "aorta", 9: "inferior_vena_cava",
    10: "portal_vein_and_splenic_vein", 11: "pancreas", 12: "adrenal_gland_right",
    13: "adrenal_gland_left", 14: "lung_right", 15: "lung_left", 16: "small_bowel",
    17: "duodenum", 18: "colon", 19: "iliac_artery_left", 20: "iliac_artery_right",
    21: "iliac_vena_left", 22: "iliac_vena_right", 23: "gluteus_maximus_left",
    24: "gluteus_maximus_right", 25: "gluteus_medius_left", 26: "gluteus_medius_right",
    27: "autochthon_left", 28: "autochthon_right", 29: "iliopsoas_left", 30: "iliopsoas_right",
    31: "hip_left", 32: "hip_right", 33: "sacrum", **_RIBS_L, **_RIBS_R, **_VERT,
}
assert len(LABELS) == 62 and max(LABELS) == 62, len(LABELS)

# the 13 abdominal organs we benchmark (matches the FLARE/AMOS CT set)
ABDO = {1: "spleen", 2: "kidney_R", 3: "kidney_L", 4: "gallbladder", 5: "liver",
        6: "esophagus", 7: "stomach", 11: "pancreas", 12: "adrenal_R", 13: "adrenal_L",
        16: "small_bowel", 17: "duodenum", 18: "colon"}
# small/hard "tail" organs the fragility premise is about
TAIL = {4, 6, 11, 12, 13, 17}  # gallbladder, esophagus, pancreas, adrenal_R/L, duodenum

# short display names for figures
SHORT = {**ABDO, 8: "aorta", 9: "IVC", 10: "portal_v", 14: "lung_R", 15: "lung_L"}


def dataset_json_labels():
    """nnU-Net v2 'labels' dict: {name: int}, background first."""
    d = {"background": 0}
    for k, v in LABELS.items():
        d[v] = k
    return d
