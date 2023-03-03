#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/pipeline_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

DATA_DIR="/home/roguz/freesound/freesound-perceptual_similarity/data"
DATASET_NAME="eval"
MODEL_NAME="audioset-yamnet_v1"

EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"

#############################################################################

for file in "$EMBED_DIR/"*; do # for each embedding dir
    f=$(basename -- "$file")   # get the basename=embed_name
    if [[ $f == "$MODEL_NAME-"* ]]; then # if the embed contains model-
        echo "======================================================================="
        echo $f
        SUFFIX="${f/$MODEL_NAME-/""}" # Strip model name to get the suffix
        yamnet-search_evaluate.sh $SUFFIX "nn"
    fi
done