#!/bin/bash

source ps/bin/activate

#############################################################################

DATA_DIR="$(pwd)/data"
DATASET_NAME="FSD50K.eval_audio"
SIMILARITY_DIR="$DATA_DIR/similarity_results/$DATASET_NAME"

#############################################################################

searches=("dot" "nn")

for file in "$SIMILARITY_DIR/"*; do # for each embedding dir
    # Check if the embedding dir contains the model name
    if [[ "$file" == *"$1"* ]]; then
        echo "$file"
        if [[ "$file" == *"fs-essentia-extractor_legacy-"* ]]; then
            # Run the script
            python evaluate.py -p=$file/nn/similarity_results.json
        elif [[ "$file" == *"audioset-vggish-3-"* ]]; then
            # Run the script for each search type
            for search in "${searches[@]}"; do
                python evaluate.py -p=$file/$search/similarity_results.json
            done
        elif [[ "$file" == *"audioset-yamnet-1-"* ]]; then
            # Run the script for each search type
            for search in "${searches[@]}"; do
                python evaluate.py -p=$file/$search/similarity_results.json
            done
        elif [[ "$file" == *"fsd-sinet-"* ]]; then
            # Determine which type of FSD-SINet model it is
            found=0
            if [[ "$file" == *"fsd-sinet-vgg41-tlpf-1-"* ]]; then
                variation="fsd-sinet-vgg41-tlpf-1"
                found=1
            elif [[ "$file" == *"fsd-sinet-vgg42-aps-1-"* ]]; then
                variation="fsd-sinet-vgg42-aps-1"
                found=1
            elif [[ "$file" == *"fsd-sinet-vgg42-tlpf_aps-1-"* ]]; then
                variation="fsd-sinet-vgg42-tlpf_aps-1"
                found=1
            elif [[ "$file" == *"fsd-sinet-vgg42-tlpf-1-"* ]]; then
                variation="fsd-sinet-vgg42-tlpf-1"
                found=1
            fi
            # Only execute the bash script if the model name is found
            if [[ "$found" == 1 ]]; then
                # Run the script for each search type
                for search in "${searches[@]}"; do
                    python evaluate.py -p=$file/$search/similarity_results.json
                done
            fi
        fi
    fi
done
