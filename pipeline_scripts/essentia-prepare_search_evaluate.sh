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

DATA_DIR="/home/roguz/freesound/freesound-perceptual_similarity/data"
DATASET_NAME="eval"
MODEL_NAME="fs-essentia-extractor_legacy"

#############################################################################

EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME/$MODEL_NAME"
SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME/$MODEL_NAME"
EVAL_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME/$MODEL_NAME"

#############################################################################

echo "======================================================================="
echo "Working with:"
echo $EMBED_DIR
echo $SIMILARITY_DIR
echo $EVAL_DIR
echo

#############################################################################

if [[ $1 == -1 ]]; then
    N=846
else
    N=$1
fi
echo "N=$N"

#############################################################################

# Prepare the embeddings
echo "======================================================================="
echo "Preparation"
python fs-essentia-extractor_legacy-embeddings_prepare.py -p=$EMBED_DIR -N=$1
EMBED_DIR="$EMBED_DIR-PCA_$N"
echo $EMBED_DIR
echo

#############################################################################

# Perform similarity search
echo "======================================================================="
echo "Similarity Search"
python similarity_search.py -p=$EMBED_DIR -s=nn
SIMILARITY_PATH="$SIMILARITY_DIR-PCA_$N/nn/similarity_results.json"
echo $SIMILARITY_PATH
echo

#############################################################################

# Evaluate
echo "======================================================================="
echo "Evaluation"
python evaluate.py -p=$SIMILARITY_PATH
echo
echo "======================================================================="

#############################################################################