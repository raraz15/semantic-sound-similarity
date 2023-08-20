#!/bin/bash

SCRIPT_DIR="$(pwd)/scripts/pipelines/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

# if [ $# == 0 ]; then
#     echo "Description: Takes extracted clap model embeddings and prepares them, 
#     searches for similarity, and performs the evaluation pipeline using the 
#     range of variables defined on this script."
#     echo "Usage: $0"
#     exit 0
# fi

MODEL_NAME="AudioCLIP-Full-Training"

#############################################################################

# Define here the variables of the experiment
variable1=(100 200 -1)
variable2=("--normalization" "--no-normalization")
variable3=("dot" "nn")

#############################################################################

for v1 in ${variable1[@]}; do
    for v2 in ${variable2[@]}; do
        for v3 in ${variable3[@]}; do
            echo "======================================================================="
            echo "Experiment Variables:"
            echo "v1=$v1, v2=$v2, v3=$v3"
            audioclip-prepare_search_evaluate.sh $v1 $v2 $v3
        done
    done
done

# Compare the results of the experiments
python code/plot_evaluation_results_comparisons.py $MODEL_NAME

#############################################################################