#!/bin/bash

source ps/bin/activate

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Takes extracted yamnet embeddings and prepares them, 
    searches for similarity, and performs the evaluation pipeline."
    echo "Usage: $0 param1 param2 param3 param4"
    echo "param1: aggregation"
    echo "param2: N_PCA"
    echo "param2: N_PCA"
    echo "param3: normalization"
    echo "param4: search type"
    exit 0
fi

#############################################################################

DATA_DIR="/home/roguz/freesound/freesound-perceptual_similarity/data"
DATASET_NAME="eval"
MODEL_NAME="audioset-yamnet_v1"

#############################################################################

EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME/$MODEL_NAME"
SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME/$MODEL_NAME"
EVAL_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME/$MODEL_NAME"

echo "======================================================================="
echo "Working with:"
echo $EMBED_DIR
echo $SIMILARITY_DIR
echo $EVAL_DIR
echo

#############################################################################

if [[ $2 == -1 ]]; then
    N=1024
else
    N=$2
fi
echo "N=$N"

if [[ $3 == "--no-normalization" ]]; then
    SUFFIX="Agg_$1-PCA_$N-Norm_False"
else
    SUFFIX="Agg_$1-PCA_$N-Norm_True"
fi
echo $SUFFIX
echo

#############################################################################

# Prepare the embeddings
echo "======================================================================="
echo "Preparation"
python audioset_yamnet_v1-embedding_prepare.py -p=$EMBED_DIR -a=$1 -N=$2 $3
PREP_EMBED_DIR="$EMBED_DIR-$SUFFIX"
echo $PREP_EMBED_DIR
echo

#############################################################################

# Perform similarity search
echo "======================================================================="
echo "Similarity Search"
python similarity_search.py -p=$PREP_EMBED_DIR -s=$4
SIMILARITY_PATH="$SIMILARITY_DIR-$SUFFIX/$4/similarity_results.json"
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