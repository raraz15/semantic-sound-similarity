#!/bin/bash

source ps/bin/activate

#############################################################################

model_names=("fs-essentia-extractor_legacy"
            "audioset-vggish-3" 
            "audioset-yamnet-1" 
            "fsd-sinet-vgg41-tlpf-1" 
            "fsd-sinet-vgg42-aps-1" 
            "fsd-sinet-vgg42-tlpf_aps-1" 
            "fsd-sinet-vgg42-tlpf-1" 
            "openl3-env-mel128-emb512-3"
    	    "openl3-env-mel256-emb512-3" 
            "openl3-env-mel256-emb6144-3"
            "clap-630k-audioset-fusion-best" 
            "clap-630k-fusion-best"
            "clap-music_speech_audioset_epoch_15_esc_89.98")

#############################################################################

for model_name in ${model_names[@]}; do
    echo $model_name
    python code/plot_evaluation_results_comparisons.py $model_name --presentation
done

#############################################################################
