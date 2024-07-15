#!/bin/bash

SCRIPT_DIR="$(pwd)/scripts/pipelines/"
DATA_DIR="$(pwd)/data"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

# if [ $# == 0 ]; then
#     echo "Description: Takes extracted and prepared CLAP model embeddings,
#     searches for similarity, and performs the evaluation pipeline using the
#     experiment variables defined inside this script."
#     echo "Usage: $0 param1"
#     exit 0
# fi

#############################################################################

# Define here the variables of the experiment
variable1=(dot nn)

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
        for v1 in ${variable1[@]}; do
            clap_2022-search_evaluate.sh $SUFFIX $v1
        done
    fi
done

# Compare the results of the experiments
python code/plot_evaluation_results_comparisons.py =$1

#############################################################################