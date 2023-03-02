#!/bin/bash

# Takes prepared embeddings, searches for similarity, and performs the evalu-
# ation pipeline.
# $1 = Prepared embedding name
# $4 = search type

source ~/.bashrc
source ps/bin/activate

#############################################################################

DATA_DIR="/home/roguz/freesound/freesound-perceptual_similarity/data"
DATASET_NAME="eval"
MODEL_NAME="audioset-yamnet_v1"

#############################################################################

EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"
SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME/$MODEL_NAME"
EVAL_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME/$MODEL_NAME"
PREP_EMBED_DIR="$EMBED_DIR/$1"

echo "======================================================================="
echo "Working with:"
echo $PREP_EMBED_DIR
echo $SIMILARITY_DIR
echo $EVAL_DIR
echo

#############################################################################

# Perform similarity search
echo "======================================================================="
echo "Similarity Search"
python similarity_search.py -p=$PREP_EMBED_DIR -s=$2
SIMILARITY_PATH="$SIMILARITY_DIR-$SUFFIX/$2/similarity_results.json"
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