#!/bin/bash

source ps/bin/activate

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Takes prepared embeddings, searches for similarity, 
    and performs the evaluation pipeline."
    echo "Usage: $0 param1 param2 param3"
    echo "param1: fsd_sinet name"
    echo "param2: suffix of prepared embedding"
    echo "param3: search_type"
    exit 0
fi

#############################################################################

MODEL_NAME=$1
DATASET_NAME="FSD50K.eval_audio"

#############################################################################

DATA_DIR="$(pwd)/data"
EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"
SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME/$MODEL_NAME"
EVAL_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME/$MODEL_NAME"
PREP_EMBED_DIR="$EMBED_DIR/$MODEL_NAME-$2"

echo "======================================================================="
echo "Input Directory:"
echo $PREP_EMBED_DIR
echo
echo "Output Directories:"
echo $SIMILARITY_DIR
echo $EVAL_DIR
echo

#############################################################################

# Perform similarity search
echo "======================================================================="
echo "Similarity Search"
python similarity_search.py -p=$PREP_EMBED_DIR -s=$3
SIMILARITY_PATH="$SIMILARITY_DIR-$2/$3/similarity_results.json"
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