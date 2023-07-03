#!/bin/bash

source ps/bin/activate

#############################################################################

DATA_DIR="$(pwd)/data"
DATASET_NAME="FSD50K.eval_audio"
SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME"

#############################################################################

searches=("dot" "nn")

for file in "$SIMILARITY_DIR/"*; do # for each similarity dir
    # Check if the embedding dir contains the model name
    if [[ "$file" == *"$1"* ]]; then
        echo "$file"
        if [[ "$file" == *"fs-essentia-extractor_legacy-"* ]]; then
            # Run the script
            python evaluate.py $file/nn/similarity_results.json #--metrics=$2
        else
            # Run the script for each search type
            for search in "${searches[@]}"; do
                python evaluate.py $file/$search/similarity_results.json #--metrics=$2
            done
        fi
    fi
done
