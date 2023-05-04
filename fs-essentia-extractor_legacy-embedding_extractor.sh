#!/bin/bash

#############################################################################

# set this to where FSD50K.eval_audio is located
DATASET_NAME="FSD50K.eval_audio"
AUDIO_DIR="/data/FSD50K/${DATASET_NAME}/"

EXTRACTOR="fs-essentia-extractor_legacy"
OUTPUT_DIR="$(pwd)/data/embeddings/${DATASET_NAME}/${EXTRACTOR}/"

#############################################################################

mkdir -p $OUTPUT_DIR

for file in "$AUDIO_DIR"*; do
    fname="$(basename $file)"
    fname="${fname%.*}"
    docker run -it --rm -v $file:"/${fname}.wav" -v $OUTPUT_DIR:/outdir $EXTRACTOR -- python main.py -i "/${fname}.wav" -o /outdir
done

echo "======================================================================="
echo "Done!"

#############################################################################