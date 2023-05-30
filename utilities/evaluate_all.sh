#!/bin/bash
source ps/bin/activate

eval_dir="/home/roguz/freesound/freesound-perceptual_similarity/data/similarity_results/FSD50K.eval_audio"

searches=("dot" "nn")
for file in "$eval_dir/"*; do # for each embedding dir

    if [[ "$file" == *"fs-essentia-extractor_legacy"* ]]; then
        results_dir="$file/nn/similarity_results.json"
        echo $results_dir
        python evaluate.py -p=$results_dir
    else
        for search in ${searches[@]}; do
            results_dir="$file/$search/similarity_results.json"
            echo $results_dir
            python evaluate.py -p=$results_dir
        done
    fi
done