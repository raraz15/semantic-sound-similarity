#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/pipeline_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

essentia-prepare_search_evaluate.sh 100
essentia-prepare_search_evaluate.sh 100
essentia-prepare_search_evaluate.sh 200
essentia-prepare_search_evaluate.sh -1

#############################################################################