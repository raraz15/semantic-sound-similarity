#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/pipeline_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

# Define here the variables of the experiment
variable1=("mean")
variable2=(100 200 -1)
variable3=("--no-normalization" "")
variable4=("dot" "nn")

#############################################################################

for v1 in ${variable1[@]}; do
    for v2 in ${variable2[@]}; do
        for idx in ${!variable3[@]}; do
            v3=${variable3[$idx]} # To deal with the "" string
            for v4 in ${variable4[@]}; do
                echo "$v1 $v2 $v3 $v4"
                yamnet-prepare_search_evaluate.sh $v1 $v2 $v3 $v4
            done
        done
    done
done

#############################################################################