#!/bin/bash

source ps/bin/activate

#############################################################################

#if [ $# == 0 ]; then
#    echo "Description: Takes extracted yamnet embeddings and prepares them, 
#    searches for similarity, and performs the evaluation pipeline."
#    echo "Usage: $0 param1"
#    echo "param1: fsd_sinet name"
#    exit 0
#fi

#############################################################################

model_names=("audioset-vggish-3" "audioset-yamnet-1" "fsd-sinet-vgg41-tlpf-1" 
"fsd-sinet-vgg42-aps-1" "fsd-sinet-vgg42-tlpf_aps-1" 
"fsd-sinet-vgg42-tlpf-1" "fs-essentia-extractor_legacy")
#############################################################################

for model_name in ${model_names[@]}; do
    echo $model_name
    python plot.py --model=$model_name
done

# Compare the results of the experiments
#python plot.py --model=$1

#############################################################################