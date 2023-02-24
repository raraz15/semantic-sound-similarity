#!/bin/bash

source ~/.bashrc
source ps/bin/activate

#############################################################################

DATA_DIR="/home/roguz/freesound/freesound-perceptual_similarity/data"
DATASET_NAME="eval"
MODEL_NAME="audioset-yamnet_v1"

#############################################################################

EMBED_DIR="${DATA_DIR}/embeddings/${DATASET_NAME}/${MODEL_NAME}"
SIMILARITY_DIR="${DATA_DIR}/similarity_results/${DATASET_NAME}/${MODEL_NAME}"
EVAL_DIR="${DATA_DIR}/evaluation_results/${DATASET_NAME}/${MODEL_NAME}"

echo "Working with:"
echo $EMBED_DIR
echo $SIMILARITY_DIR
echo $EVAL_DIR
echo

#############################################################################

S="dot"

if [ $2 -eq -1 ]; then
    N=1024
else
    N=$2
fi
echo "N=${N}"

if [ $3 == "--no-normalization" ]; then
    SUFFIX="Agg_${1}-PCA_${N}-Norm_False"
else
    SUFFIX="Agg_${1}-PCA_${N}-Norm_True"
fi
echo $SUFFIX

#############################################################################

# Prepare the embeddings
echo "Preparation"
python prepare_yamnet_embeddings.py -p=$EMBED_DIR -a=$1 -N=$2 $3
echo

#############################################################################

# Perform similarity search
echo "Similarity Search"
EMBED_DIR="${EMBED_DIR}-${SUFFIX}"
echo $EMBED_DIR
python similarity_search.py -p=$EMBED_DIR -s=$S
echo

#############################################################################

# Evaluate
echo "Evaluation"
SIMILARITY_DIR="${SIMILARITY_DIR}-${SUFFIX}"
SIMILARITY_PATH="${SIMILARITY_DIR}/${S}-results.json"
echo $SIMILARITY_PATH
python evaluate.py -p=$SIMILARITY_PATH
echo

#############################################################################