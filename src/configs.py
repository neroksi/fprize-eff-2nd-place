import torch
from pathlib import Path
import numpy as np
import transformers
import os


transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

USE_AMP = True

INTER_GROUP_LR_SCALE = 100.

FOLD_COL_NAME = "gp_essay_strat_eff" #

LOSS_NAME = "ce" # 

IS_PSL = False

TRUE_OBS_WEIGHT_RATIO = 0.8
USE_SAMPLE_WEIGHTS = False

SIGMOID_COMPATIBLE_LOSSES = ["bce",]
SOFTMAX_COMPATIBLE_LOSSES = ["ce", ]

SAVE_METRIC_NAME = "loss_v2_val"
SAVE_MODE = "min"
SAVE_PREFIX = "fpeff"

REMOVE_LEADING_PADS = True

PROJECT_ROOT = Path(__file__).resolve().absolute().parent.parent

MODEL_ROOT = PROJECT_ROOT / "models"
DATA_ROOT = PROJECT_ROOT / "data"
COMP_DATA_ROOT = DATA_ROOT / "feedback-prize-effectiveness"

TRAIN_CSV_PATH = COMP_DATA_ROOT / "train.csv"
SAMPLE_SUB_CSV_PATH = COMP_DATA_ROOT / "sample_submission.csv"
TEST_CSV_PATH = COMP_DATA_ROOT / "test.csv"
TRAIN_ROOT = COMP_DATA_ROOT / "train"
TEST_ROOT = COMP_DATA_ROOT / "test"


# USE_CLASS_WEIGHTS = True
# CLASS_WEIGHTS = [2.51170864, 0.1201229 , 0.36816846]
# CLASS_WEIGHTS_USE_TIME_RATIO = 1.0 # Class weights will become the same after this time ratio passed

MAXLEN = 256

SEED = 321

MODEL_NAME = "roberta-base"

TRAIN_BATCH_SIZE = 2
TRAIN_NUM_WORKERS = 2

VAL_BATCH_SIZE = 2
VAL_NUM_WORKERS = 2

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

CLIP_GRAD_NORM = 2.0


OPTIMIZER_LR = 8e-6
OPTIMIZER_WEIGHT_DECAY = 0.01 #1e-5
SCHEDULER_ETA_MIN = 5e-7
WARMUP_RATIO = 0.25
SCHEDULER_GAMMA = 3

P_MASK_SIZE_LOW = 0.10
P_MASK_SIZE_HIGH = 0.30
P_MASK_FREQ = 0.80


P_RANDOM_START = 0.80 #0.50
P_START_AT_SEQ_BEGINNING = .80 # 0.80 # Prob to start at beginning if not random start
MIN_SEQ_LEN = 4096 #512 # All sequences longer than this could be truncated
FORCE_TRUNC_FREQ = 0.0

PYTORCH_CE_IGNORE_INDEX = -100

# STRIDE_MAX_LEN_RATIO = 2

P_DROPS = None
N_DROPS = 5

NEW_LINE_TOKEN = "<br>"

START_SPECIAL_TOKEN_FORMAT = "<{}>"
END_SPECIAL_TOKEN_FORMAT = "</{}>"

D_SPAN_TOKEN_FORMAT = "<span::{:02d}>"
MAX_OVER_ALL_NUM_SPANS = 25 # max overall allowable spans per essay
MAX_NUM_SPANS = None

ESSAY_START_TOKEN = "<essay>"
ESSAY_END_TOKEN = "</essay>"

TRUNCATION_TOKEN = "<trunc...>"

NULL_DISCOURSE_START_TOKEN = "<none>"
NULL_DISCOURSE_END_TOKEN = "</none>"
NULL_DISCOURSE_TYPE = "__out__"

D_SPAN_TOKENS = [D_SPAN_TOKEN_FORMAT.format(i) for i in  range(MAX_OVER_ALL_NUM_SPANS)]
D_SPAN_TOKEN_IDS = {}

SPECIAL_TOKENS = [
    '<lead>',
    '</lead>',
    '<position>',
    '</position>',
    '<claim>',
    '</claim>',
    '<evidence>',
    '</evidence>',
    '<counterclaim>',
    '</counterclaim>',
    '<rebuttal>',
    '</rebuttal>',
    '<concluding-statement>',
    '</concluding-statement>',
    NEW_LINE_TOKEN,

    ESSAY_START_TOKEN,
    ESSAY_END_TOKEN,

    TRUNCATION_TOKEN,

    NULL_DISCOURSE_START_TOKEN,
    NULL_DISCOURSE_END_TOKEN,

    *D_SPAN_TOKENS,
]

CLASS2ID = {
    "Ineffective": 0,
    "Adequate": 1,
    "Effective": 2,
}

ID2CLASS = {v: k for k,v in CLASS2ID.items()}
NUM_TARGETS = len(CLASS2ID)
TARGET_NAMES = [ID2CLASS[i] for i in range(NUM_TARGETS)]


DISCOURSE_BFORMAT = "B-{}"
DISCOURSE_IFORMAT = "I-{}"
DISCOURSE_EFORMAT = "E-{}"

DISCOURSE_PAD_ID = 0
DISCOURSE_OUT_ID = 1

D2ID = {
    "__pad__": DISCOURSE_PAD_ID,

    NULL_DISCOURSE_TYPE: DISCOURSE_OUT_ID,

    'B-Lead': 2,
    'I-Lead': 3,
    'E-Lead': 4,

    'B-Position': 5,
    'I-Position': 6,
    'E-Position': 7,

    'B-Claim': 8,
    'I-Claim': 9,
    'E-Claim': 10,

    'B-Counterclaim': 11,
    'I-Counterclaim': 12,
    'E-Counterclaim': 13,

    'B-Rebuttal': 14,
    'I-Rebuttal': 15,
    'E-Rebuttal': 16,

    'B-Evidence': 17,
    'I-Evidence': 18,
    'E-Evidence': 19,

    'B-Concluding Statement': 20,
    'I-Concluding Statement': 21,
    'E-Concluding Statement': 22,

}

ID2D = {id_: discourse for discourse, id_ in D2ID.items()}

NUM_D = len(D2ID)