#!/bin/bash

source ps/bin/activate

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Takes prepared embeddings, searches for similarity, 
    and performs the evaluation pipeline."
    echo "Usage: $0 param1"
    echo "param1: suffix of prepared embedding"
    exit 0
fi

#############################################################################

MODEL_NAME="fs-essentia-extractor_legacy"
DATASET_NAME="FSD50K.eval_audio"
EMBED_NAME="$MODEL_NAME-$1"

#############################################################################

DATA_DIR="$(pwd)/data"
EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"
PREP_EMBED_DIR="$EMBED_DIR/$EMBED_NAME"

SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME/$EMBED_NAME/nn"
EVAL_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME/$EMBED_NAME/nn"

echo "======================================================================="
echo "Input Directory:"
echo $PREP_EMBED_DIR
echo
echo "Output Directories:"
echo $SIMILARITY_DIR
echo $EVAL_DIR
echo "======================================================================="
echo

#############################################################################
# Perform similarity search

echo "Similarity Search"
python code/similarity_search.py $PREP_EMBED_DIR -s=nn
SIMILARITY_PATH="$SIMILARITY_DIR/similarity_results.json"
echo "======================================================================="
echo

#############################################################################
# Evaluate

echo "Evaluation"
python code/evaluate_map_at_n.py $SIMILARITY_PATH
echo "======================================================================="
echo

#############################################################################