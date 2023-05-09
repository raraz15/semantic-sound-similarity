#!/bin/bash

SCRIPT_DIR="$(pwd)/pipeline_scripts/"
DATA_DIR="$(pwd)/data"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

DATASET_NAME="FSD50K.eval_audio"
MODEL_NAME="audioset-yamnet-1"
EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"

#############################################################################

for file in "$EMBED_DIR/"*; do # for each embedding dir
    f=$(basename -- "$file")   # get the basename=embed_name
    if [[ $f == "$MODEL_NAME-"* ]]; then # if the embed contains model-
        echo "======================================================================="
        echo $f
        SUFFIX="${f/$MODEL_NAME-/""}" # Strip model name to get the suffix
        yamnet-search_evaluate.sh $SUFFIX
    fi
done

# Compare the results of the experiments
python plot.py --model=audioset-yamnet-1

#############################################################################