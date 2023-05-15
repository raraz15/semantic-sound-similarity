#!/bin/bash

source ps/bin/activate

#############################################################################

model_names=("audioset-vggish-3" "audioset-yamnet-1" "fsd-sinet-vgg41-tlpf-1" 
"fsd-sinet-vgg42-aps-1" "fsd-sinet-vgg42-tlpf_aps-1" 
"fsd-sinet-vgg42-tlpf-1" "fs-essentia-extractor_legacy")

#############################################################################

for model_name in ${model_names[@]}; do
    echo $model_name
    python plot.py --model=$model_name
done

#############################################################################