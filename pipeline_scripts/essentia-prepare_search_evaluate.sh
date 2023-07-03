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
EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME/$MODEL_NAME"
SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME/$MODEL_NAME"
EVAL_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME/$MODEL_NAME"

echo "======================================================================="
echo "Input Directory:"
echo $EMBED_DIR
echo

#############################################################################

# Deal with No PCA case
if [[ $1 == -1 ]]; then
    N=846
else
    N=$1
fi
PREP_EMBED_DIR="$EMBED_DIR-PCA_$N"

echo "Output Directories:"
echo $PREP_EMBED_DIR
echo $SIMILARITY_DIR
echo $EVAL_DIR

#############################################################################

# Prepare the embeddings
echo "======================================================================="
echo "Preparation"
python fs-essentia-extractor_legacy-create_clip_level_embedding.py $EMBED_DIR -N=$1
echo $PREP_EMBED_DIR
echo

#############################################################################

# Perform similarity search
echo "======================================================================="
echo "Similarity Search"
python similarity_search.py $PREP_EMBED_DIR -s=nn
SIMILARITY_PATH="$SIMILARITY_DIR-PCA_$N/nn/similarity_results.json"
echo $SIMILARITY_PATH
echo

#############################################################################

# Evaluate
echo "======================================================================="
echo "Evaluation"
python evaluate.py $SIMILARITY_PATH
echo
echo "======================================================================="

#############################################################################