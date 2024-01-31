j#!/bin/bash

SCRIPT_DIR="$(pwd)/scripts/pipelines/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Runs scripts/pipelines/imagebind-prepare_search_evaluate.sh
    for each combination of the variables defined on this script."
    echo "Usage: $0 param1"
    echo "param1: ImageBind name"
    exit 0
fi

#############################################################################

# Define here the variables of the experiment
variable1=(100 512 -1)
variable2=("--no-normalization" "--normalization")
variable3=("dot" "nn")

#############################################################################

for v1 in ${variable1[@]}; do
    for v2 in ${variable2[@]}; do
        for v3 in ${variable3[@]}; do
            echo "======================================================================="
            echo "Experiment Variables:"
            echo "v1=$v1, v2=$v2, v3=$v3"
            imagebind-prepare_search_evaluate.sh $1 $v1 $v2 $v3
        done
    done
done

# Compare the results of the experiments
python code/plot_evaluation_results_comparisons.py $1

#############################################################################