#!/bin/bash

source ps/bin/activate

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Takes extracted yamnet embeddings and prepares them, 
    searches for similarity, and performs the evaluation pipeline."
    echo "Usage: $0 param1 param2 param3 param4 param5"
    echo "param1: fsd_sinet name"
    echo "param2: aggregation"
    echo "param3: N_PCA"
    echo "param4: normalization"
    echo "param5: search type"
    exit 0
fi

#############################################################################

DATA_DIR="/home/roguz/freesound/freesound-perceptual_similarity/data"
DATASET_NAME="FSD50K.eval_audio"

#############################################################################

EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME/$1"
SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME/$1"
EVAL_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME/$1"

echo "======================================================================="
echo "Working with:"
echo $EMBED_DIR
echo $SIMILARITY_DIR
echo $EVAL_DIR
echo

#############################################################################

if [[ $3 == -1 ]]; then
    if [[ $1 == "fsd-sinet-vgg41-tlpf-1" ]]; then
        N=256
    elif [[ $1 == "fsd-sinet-vgg42-aps-1" ]]; then
        N=512
    elif [[ $1 == "fsd-sinet-vgg42-tlpf_aps-1" ]]; then
        N=512
    elif [[ $1 == "fsd-sinet-vgg42-tlpf-1" ]]; then
        N=512
    else
        echo "Wrong FSD-SINet name"
        exit 1
    fi
else
    N=$3
fi
echo "N=$N"

if [[ $4 == "--no-normalization" ]]; then
    SUFFIX="Agg_$2-PCA_$N-Norm_False"
else
    SUFFIX="Agg_$2-PCA_$N-Norm_True"
fi
echo $SUFFIX
echo

#############################################################################

# Prepare the embeddings
echo "======================================================================="
echo "Preparation"
python prepare_embeddings.py -p=$EMBED_DIR -a=$2 -N=$3 $4
PREP_EMBED_DIR="$EMBED_DIR-$SUFFIX"
echo $PREP_EMBED_DIR
echo

#############################################################################

# Perform similarity search
echo "======================================================================="
echo "Similarity Search"
python similarity_search.py -p=$PREP_EMBED_DIR -s=$5
SIMILARITY_PATH="$SIMILARITY_DIR-$SUFFIX/$5/similarity_results.json"
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