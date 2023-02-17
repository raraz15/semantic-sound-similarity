#!/bin/bash

source ~/.bashrc

AUDIO_DIR="/data/FSD50K/FSD50K.eval_audio/"
OUTPUT_DIR="/home/roguz/freesound-perceptual_similarity/embeddings/fs-essentia-extractor_v1/eval/"
mkdir -p $OUTPUT_DIR

# TODO: save all frames ?
for file in "$AUDIO_DIR"*; do
    fname="$(basename $file)"
    fname="${fname%.*}"
    docker run -it --rm -v $file:"/${fname}.wav" -v $OUTPUT_DIR:/outdir fs-essentia-extractor_v1 -- python main.py -i "/${fname}.wav" -o /outdir
done