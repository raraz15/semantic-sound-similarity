#!/bin/bash

SCRIPT_DIR="$(pwd)/pipeline_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

DATA_DIR="$(pwd)/data"
DATASET_NAME="FSD50K.eval_audio"
MODEL_NAME="fs-essentia-extractor_legacy"

EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"

#############################################################################
for file in "$EMBED_DIR/"*; do # for each embedding dir
    f=$(basename -- "$file")   # get the basename=embed_name
    if [[ $f == "$MODEL_NAME-"* ]]; then # if the embed contains model-
        echo "======================================================================="
        echo $f
        readarray -d - -t strarr <<< $f # Split from -
        SUFFIX="${strarr[3]}"     # 3rd is the PCA for essentia
        essentia-search_evaluate.sh $SUFFIX "nn"
    fi
done

#############################################################################