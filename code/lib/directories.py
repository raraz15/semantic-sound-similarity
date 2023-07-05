import os

# These directories should point to the FSD50K dataset
######################################################
FSD50K_DIR = "/data/FSD50K"
AUDIO_DIR = f"{FSD50K_DIR}/FSD50K.eval_audio"
GT_PATH = f"{FSD50K_DIR}/FSD50K.ground_truth/eval.csv"
######################################################

# These directories automatically point to the code
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_DIR = os.path.join(PROJECT_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
ANALYSIS_DIR = os.path.join(DATA_DIR, "similarity_results")
EVAL_DIR = os.path.join(DATA_DIR, "evaluation_results")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")