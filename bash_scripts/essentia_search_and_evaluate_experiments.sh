#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/bash_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

essentia_search_and_evaluate.sh "PCA_100" "nn"
essentia_search_and_evaluate.sh "PCA_200" "nn"
essentia_search_and_evaluate.sh "PCA_846" "nn"