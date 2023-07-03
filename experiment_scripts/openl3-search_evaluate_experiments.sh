#!/bin/bash

SCRIPT_DIR="$(pwd)/pipeline_scripts/"
DATA_DIR="$(pwd)/data"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Takes extracted yamnet embeddings and prepares them, 
    searches for similarity, and performs the evaluation pipeline."
    echo "Usage: $0 param1"
    echo "param1: openl3 name"
    exit 0
fi

#############################################################################

DATASET_NAME="FSD50K.eval_audio"
EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"

#############################################################################

for file in "$EMBED_DIR/"*; do # for each embedding dir
    f=$(basename -- "$file")   # get the basename=embed_name
    if [[ $f == "$1-"* ]]; then # if the embed contains model-
        echo "======================================================================="
        echo $f
        SUFFIX="${f/$1-/""}" # Strip model name to get the suffix
        openl3-search_evaluate.sh $SUFFIX
    fi
done

# Compare the results of the experiments
python plot_evaluation_results.py =$1


#############################################################################