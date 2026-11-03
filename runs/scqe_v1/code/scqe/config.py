"""SCQE frozen protocol constants (scqe_experiment_v1, 2026-10-12)."""
from pathlib import Path
import os

ROOT = Path(os.environ.get("PF_RPN_SR_ROOT", "/home/root-user/projects/PF_RPN_SR"))
RUN_DIR = ROOT / "runs" / "scqe_v1"

PF_RPN_DIR = ROOT / "third_party" / "PF-RPN"
CDFSOD_DIR = ROOT / "third_party" / "CDFSOD-benchmark"
CHECKPOINT_PATH = ROOT / "checkpoints" / "pf_rpn_swinb_5p_coco_imagenet.pth"
CONFIG_PATH = PF_RPN_DIR / "configs" / "pf-rpn" / "pf-rpn_coco-imagenet.py"

DATA_DIR = ROOT / "data"
DIOR_DIR = DATA_DIR / "DIOR"
C0_DATASETS = ROOT / "c0_datasets"

# Structural settings (Table: fixed design defaults)
Q = 900
ETA = 0.5                     # protected fraction
P_PROBES = 16                 # VJP probes
M_MAX = 32                    # exchange cap
L_IN = 128                    # incoming shortlist
L_OUT = 32                    # outgoing shortlist
GRID = 7                      # sampling grid n_g
BETA = 1.5                    # context expansion
EPS_NORM = 1e-6               # normalization threshold
NG = GRID * GRID

# Seeds
SEED_GLOBAL = 20261012
SEED_BOOTSTRAP = 20261013
SEED_PROBE_BASE = "20261012"  # probe seed string suffix
SEED_SUPPORT_CAL = [2023, 42]
SEED_SUPPORT_PILOT = [2023, 42, 1234]
SEED_SUPPORT_ALL = [2023, 42, 1234, 7, 99]

# Native scoring profile (pinned PF-RPN config)
USE_CONF_BRANCH = True
RM_LANG_EMBED = True
OUTPUT_SCORE = "both"

DIOR_CLASSES = (
    "Expressway-Service-area", "Expressway-toll-station", "airplane",
    "airport", "baseballfield", "basketballcourt", "bridge", "chimney",
    "dam", "golffield", "groundtrackfield", "harbor", "overpass", "ship",
    "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle",
    "windmill",
)

DIOR_IMG_SUBDIR_TEST = "test/new_test"
DIOR_IMG_SUBDIR_TRAIN = "train/new_train"
