#!/bin/bash

source ps/bin/activate

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Takes extracted Wav2CLIP model embeddings and prepares 
    them, searches for similarity, and performs the evaluation pipeline."
    echo "Usage: $0 param1 param2 param3"
    echo "param1: N_PCA (integer)"
    echo "param2: --no-normalization or --normalization"
    echo "param3: search type (dot or nn)"
    exit 0
fi

#############################################################################

MODEL_NAME="Wav2CLIP"
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
    N=512
else
    N=$1
fi
# There is no Aggregation for CLAP
if [[ $2 == "--no-normalization" ]]; then
    SUFFIX="Agg_none-PCA_$N-Norm_False"
else
    SUFFIX="Agg_none-PCA_$N-Norm_True"
fi
EMBED_NAME="$MODEL_NAME-$SUFFIX"

#############################################################################

EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME/$EMBED_NAME"
SIMILARITY_DIR="$DATA_DIR/similarity_rankings/$DATASET_NAME/$EMBED_NAME"
EVAL_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME/$EMBED_NAME"

echo "Output Directories:"
echo $EMBED_DIR
echo $SIMILARITY_DIR
echo $EVAL_DIR

#############################################################################

# Prepare the embeddings
echo "======================================================================="
echo "Preparation"
python code/create_clip_level_embedding.py $RAW_EMBED_DIR -a=none -N=$1 $2
echo $EMBED_DIR
echo

#############################################################################

# Perform similarity search
echo "======================================================================="
echo "Similarity Search"
python code/similarity_search.py $EMBED_DIR -s=$3
SIMILARITY_PATH="$SIMILARITY_DIR/$3/similarity_results.json"
echo $SIMILARITY_PATH
echo

#############################################################################

# Evaluate
echo "======================================================================="
echo "Evaluation"
python code/evaluate_map_at_n.py $SIMILARITY_PATH
echo
echo "======================================================================="

#############################################################################