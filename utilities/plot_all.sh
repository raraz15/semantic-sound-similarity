#!/bin/bash

source ps/bin/activate

#############################################################################

model_names=("fs-essentia-extractor_legacy" "audioset-vggish-3" 
            "audioset-yamnet-1" "fsd-sinet-vgg41-tlpf-1" 
            "fsd-sinet-vgg42-aps-1" "fsd-sinet-vgg42-tlpf_aps-1" 
            "fsd-sinet-vgg42-tlpf-1")

#############################################################################

for model_name in ${model_names[@]}; do
    echo $model_name
    python plot_evaluation_results.py =$model_name
done

#############################################################################