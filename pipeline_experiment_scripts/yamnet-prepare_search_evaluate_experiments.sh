#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/pipeline_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

yamnet-prepare_search_evaluate.sh "mean" 100 "" "dot"
yamnet-prepare_search_evaluate.sh "mean" 100 "--no-normalization" "dot"
yamnet-prepare_search_evaluate.sh "mean" 200 "" "dot"
yamnet-prepare_search_evaluate.sh "mean" 200 "--no-normalization" "dot"
yamnet-prepare_search_evaluate.sh "mean" -1 "" "dot"
yamnet-prepare_search_evaluate.sh "mean" -1 "--no-normalization" "dot"

yamnet-prepare_search_evaluate "mean" 100 "" "nn"
yamnet-prepare_search_evaluate "mean" 100 "--no-normalization" "nn"
yamnet-prepare_search_evaluate "mean" 200 "" "nn"
yamnet-prepare_search_evaluate "mean" 200 "--no-normalization" "nn"
yamnet-prepare_search_evaluate "mean" -1 "" "nn"
yamnet-prepare_search_evaluate "mean" -1 "--no-normalization" "nn"