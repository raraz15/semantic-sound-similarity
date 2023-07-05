#!/bin/bash

SCRIPT_DIR="$(pwd)/scripts/pipelines/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

DATA_DIR="$(pwd)/data"
DATASET_NAME="FSD50K.eval_audio"
EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"

#############################################################################

searches=("dot" "nn")
for file in "$EMBED_DIR/"*; do # for each embedding dir
    # Check if the embedding dir contains the model name
    if [[ "$file" == *"$1"* ]]; then
        echo "$file"
        if [[ "$file" == *"fs-essentia-extractor_legacy-"* ]]; then
            # Get the suffix from the path name
            suffix=$(echo "$file" | sed -e "s/.*fs-essentia-extractor_legacy-//")
            echo "$suffix"
            # Run the script
            essentia-search_evaluate.sh "$suffix"
        elif [[ "$file" == *"audioset-vggish-3-"* ]]; then
            # Get the suffix from the path name
            suffix=$(echo "$file" | sed -e "s/.*audioset-vggish-3-//")
            echo "$suffix"
            # Run the script for each search type
            for search in "${searches[@]}"; do
                vggish-search_evaluate.sh "$suffix" "$search"
            done
        elif [[ "$file" == *"audioset-yamnet-1-"* ]]; then
            # Get the suffix from the path name
            suffix=$(echo "$file" | sed -e "s/.*audioset-yamnet-1-//")
            echo "$suffix"
            # Run the script for each search type
            for search in "${searches[@]}"; do
                yamnet-search_evaluate.sh "$suffix" "$search"
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
                echo "variation: $variation"
                suffix=$(echo "$file" | sed -e "s/.*$variation-//")
                echo "suffix: $suffix"
                # Run the script for each search type
                for search in "${searches[@]}"; do
                    sinet-search_evaluate.sh "$variation" "$suffix" "$search"
                done
            fi
        fi
    fi
done
