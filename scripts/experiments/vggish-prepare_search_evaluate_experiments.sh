#!/bin/bash

SCRIPT_DIR="$(pwd)/scripts/pipelines/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

# Define here the variables of the experiment
variable1=("max" "median") 
variable2=(64 100 -1)
variable3=("--no-normalization" "--normalization")
variable4=("dot" "nn")

#############################################################################

for v1 in ${variable1[@]}; do
    for v2 in ${variable2[@]}; do
        for v3 in ${variable3[@]}; do
            for v4 in ${variable4[@]}; do
                echo "Experiment Variables:"
                echo "v1=$v1, v2=$v2, v3=$v3, v4=$v4"
                vggish-prepare_search_evaluate.sh $v1 $v2 $v3 $v4
            done
        done
    done
done

# Compare the results of the experiments
python code/plot_evaluation_results_comparisons.py =audioset-vggish-3

#############################################################################