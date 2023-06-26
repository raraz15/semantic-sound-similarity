#!/bin/bash

SCRIPT_DIR="$(pwd)/pipeline_scripts/"
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

# Define here the variables of the experiment
variable1=("mean")
variable2=(100 200 -1)
variable3=("--no-normalization" "--normalization")
variable4=("dot" "nn")

#############################################################################

for v1 in ${variable1[@]}; do
    for v2 in ${variable2[@]}; do
        for v3 in ${variable3[@]}; do
            for v4 in ${variable4[@]}; do
                echo "======================================================================="
                echo "Experiment Variables:"
                echo "v1=$v1, v2=$v2, v3=$v3, v4=$v4"
                openl3-prepare_search_evaluate.sh $1 $v1 $v2 $v3 $v4
            done
        done
    done
done

# Compare the results of the experiments
python plot_evaluation_results.py --model=$1

#############################################################################