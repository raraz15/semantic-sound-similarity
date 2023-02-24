#!/bin/bash

SCRIPT="/home/roguz/freesound/freesound-perceptual_similarity/bash_scripts/"
export PATH="$SCRIPT:$PATH"

#############################################################################

yamnet_model_output_to_evaluation.sh "mean" 100 "" "dot"
yamnet_model_output_to_evaluation.sh "mean" 100 "--no-normalization" "dot"
yamnet_model_output_to_evaluation.sh "mean" 200 "" "dot"
yamnet_model_output_to_evaluation.sh "mean" 200 "--no-normalization" "dot"
yamnet_model_output_to_evaluation.sh "mean" -1 "" "dot"
yamnet_model_output_to_evaluation.sh "mean" -1 "--no-normalization" "dot"

yamnet_model_output_to_evaluation.sh "mean" 100 "" "nn"
yamnet_model_output_to_evaluation.sh "mean" 100 "--no-normalization" "nn"
yamnet_model_output_to_evaluation.sh "mean" 200 "" "nn"
yamnet_model_output_to_evaluation.sh "mean" 200 "--no-normalization" "nn"
yamnet_model_output_to_evaluation.sh "mean" -1 "" "nn"
yamnet_model_output_to_evaluation.sh "mean" -1 "--no-normalization" "nn"