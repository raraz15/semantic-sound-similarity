#!/bin/bash
source ps/bin/activate

eval_dir="/home/roguz/freesound/freesound-perceptual_similarity/data/similarity_results/FSD50K.eval_audio"

searches=("dot" "nn")
for file in "$eval_dir/"*; do # for each embedding dir
    if [[ "$file" == *"audioset-yamnet-1"* ]]; then
        results_dir="$file/nn/similarity_results.json"
        echo $results_dir
        python evaluate.py -p=$results_dir
    fi
done