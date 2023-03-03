#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/bash_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

essentia_pipeline.sh 100 "nn"
essentia_pipeline.sh 100 "dot"
essentia_pipeline.sh 200 "nn"
essentia_pipeline.sh -1 "nn"

#############################################################################