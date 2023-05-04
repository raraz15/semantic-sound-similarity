#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/pipeline_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

if [ $# == 0 ]; then
    echo "Description: Takes extracted yamnet embeddings and prepares them, 
    searches for similarity, and performs the evaluation pipeline."
    echo "Usage: $0 param1"
    echo "param1: fsd_sinet name"
    exit 0
fi

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
                echo "Experiment Variables:"
                echo "v1 = $v1, v2 = $v2, v3 = $v3, v4 = $v4"
                if [[ $v3 == "" ]]; then
                    sinet-prepare_search_evaluate.sh $1 $v1 $v2 "" $v4
                else
                    sinet-prepare_search_evaluate.sh $1 $v1 $v2 $v3 $v4
                fi
            done
        done
    done
done

#############################################################################