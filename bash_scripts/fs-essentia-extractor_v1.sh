#!/bin/bash

source ~/.bashrc

EXTRACTOR="fs-essentia-extractor_v1"
AUDIO_DIR="/data/FSD50K/FSD50K.eval_audio/"
OUTPUT_DIR="/home/roguz/freesound-perceptual_similarity/data/embeddings/${EXTRACTOR}/eval/"
mkdir -p $OUTPUT_DIR

for file in "$AUDIO_DIR"*; do
    fname="$(basename $file)"
    fname="${fname%.*}"
    docker run -it --rm -v $file:"/${fname}.wav" -v $OUTPUT_DIR:/outdir $EXTRACTOR -- python main.py -i "/${fname}.wav" -o /outdir
done