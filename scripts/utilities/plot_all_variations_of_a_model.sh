#!/bin/bash

SCRIPT_DIR="$(pwd)/scripts/pipelines/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

DATA_DIR="$(pwd)/data"
DATASET_NAME="FSD50K.eval_audio"
RESULTS_DIR="$DATA_DIR/evaluation_results/$DATASET_NAME"

#############################################################################
for feat_dir in "$RESULTS_DIR/"*; do # for each feature extractor dir

    # Check if the result dir contains the model name
    if [[ "$feat_dir" == *"$1"* ]]; then
        echo "$feat_dir"

        if [[ "$feat_dir" == *"fs-essentia-extractor_legacy-"* ]]; then
            # Get the suffix from the path name
            suffix=$(echo "$feat_dir" | sed -e "s/.*fs-essentia-extractor_legacy-//")
            echo "$suffix"
        elif [[ "$feat_dir" == *"audioset-vggish-3-"* ]]; then
            # Get the suffix from the path name
            suffix=$(echo "$feat_dir" | sed -e "s/.*audioset-vggish-3-//")
            echo "$suffix"
        elif [[ "$feat_dir" == *"audioset-yamnet-1-"* ]]; then
            # Get the suffix from the path name
            suffix=$(echo "$feat_dir" | sed -e "s/.*audioset-yamnet-1-//")
            echo "$suffix"
        elif [[ "$feat_dir" == *"fsd-sinet-"* ]]; then
            # Determine which type of FSD-SINet model it is
            found=0
            if [[ "$feat_dir" == *"fsd-sinet-vgg41-tlpf-1-"* ]]; then
                variation="fsd-sinet-vgg41-tlpf-1"
                found=1
            elif [[ "$feat_dir" == *"fsd-sinet-vgg42-aps-1-"* ]]; then
                variation="fsd-sinet-vgg42-aps-1"
                found=1
            elif [[ "$feat_dir" == *"fsd-sinet-vgg42-tlpf_aps-1-"* ]]; then
                variation="fsd-sinet-vgg42-tlpf_aps-1"
                found=1
            elif [[ "$feat_dir" == *"fsd-sinet-vgg42-tlpf-1-"* ]]; then
                variation="fsd-sinet-vgg42-tlpf-1"
                found=1
            fi
            # Only execute the bash script if the model name is found
            if [[ "$found" == 1 ]]; then
                echo "variation: $variation"
                suffix=$(echo "$feat_dir" | sed -e "s/.*$variation-//")
                echo "suffix: $suffix"
                # Run the script for each search type
                for search in "${searches[@]}"; do
                    sinet-search_evaluate.sh "$variation" "$suffix" "$search"
                done
            fi
        fi

        # Plot the MAP@N for each search type
        for search_dir in "$feat_dir/"*; do
            search=$(basename -- "$search_dir")
            plot_map_at_n_single_variation.py $1 $suffix $search
        done

    fi

done
