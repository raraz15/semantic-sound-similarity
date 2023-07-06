#!/bin/bash

source ps/bin/activate

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Takes extracted essentia embeddings and prepares them, 
    searches for similarity, and performs the evaluation pipeline."
    echo "Usage: $0 param1"
    echo "param1: N_PCA"
    exit 0
fi

#############################################################################

MODEL_NAME="fs-essentia-extractor_legacy"
DATASET_NAME="FSD50K.eval_audio"

#############################################################################

DATA_DIR="$(pwd)/data"
RAW_EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME/$MODEL_NAME"

echo "======================================================================="
echo "Input Directory:"
echo $RAW_EMBED_DIR
echo

#############################################################################
# Determine the embedding name

# Deal with No PCA case
if [[ $1 == -1 ]]; then
    N=846
else
    N=$1
fi
EMBED_NAME="$MODEL_NAME-PCA_$N"

#############################################################################

EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME/$EMBED_NAME"
SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME/$EMBED_NAME"
EVAL_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME/$EMBED_NAME"

echo "Output Directories:"
echo $EMBED_DIR
echo $SIMILARITY_DIR
echo $EVAL_DIR

#############################################################################

# Prepare the embeddings
echo "======================================================================="
echo "Preparation"
python code/fs-essentia-extractor_legacy-create_clip_level_embedding.py $RAW_EMBED_DIR -N=$1
echo $EMBED_DIR
echo

#############################################################################

# Perform similarity search
echo "======================================================================="
echo "Similarity Search"
python code/similarity_search.py $EMBED_DIR -s=nn
SIMILARITY_PATH="$SIMILARITY_DIR/nn/similarity_results.json"
echo $SIMILARITY_PATH
echo

#############################################################################

# Evaluate
echo "======================================================================="
echo "Evaluation"
python code/evaluate.py $SIMILARITY_PATH
echo
echo "======================================================================="

#############################################################################