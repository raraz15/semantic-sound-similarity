#!/bin/bash

SCRIPT_DIR="$(pwd)/scripts/pipelines/"
DATA_DIR="$(pwd)/data"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

DATASET_NAME="FSD50K.eval_audio"
MODEL_NAME="audioset-yamnet-1"
EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"

#############################################################################

# Define here the variables of the experiment
variable1=(dot nn)

#############################################################################

for file in "$EMBED_DIR/"*; do # for each embedding dir
    f=$(basename -- "$file")   # get the basename=embed_name
    if [[ $f == "$MODEL_NAME-"* ]]; then # if the embed contains model-
        echo "======================================================================="
        echo $f
        SUFFIX="${f/$MODEL_NAME-/""}" # Strip model name to get the suffix
        for v1 in ${variable1[@]}; do
            yamnet-search_evaluate.sh $1 $SUFFIX $v1
        done
    fi
done

# Compare the results of the experiments
python code/plot_evaluation_results_comparisons.py =audioset-yamnet-1

#############################################################################