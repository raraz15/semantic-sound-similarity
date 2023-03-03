#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/bash_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

yamnet_pipeline.sh "mean" 100 "" "dot"
yamnet_pipeline.sh "mean" 100 "--no-normalization" "dot"
yamnet_pipeline.sh "mean" 200 "" "dot"
yamnet_pipeline.sh "mean" 200 "--no-normalization" "dot"
yamnet_pipeline.sh "mean" -1 "" "dot"
yamnet_pipeline.sh "mean" -1 "--no-normalization" "dot"

yamnet_pipeline.sh "mean" 100 "" "nn"
yamnet_pipeline.sh "mean" 100 "--no-normalization" "nn"
yamnet_pipeline.sh "mean" 200 "" "nn"
yamnet_pipeline.sh "mean" 200 "--no-normalization" "nn"
yamnet_pipeline.sh "mean" -1 "" "nn"
yamnet_pipeline.sh "mean" -1 "--no-normalization" "nn"