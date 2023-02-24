#!/bin/bash

SCRIPT="/home/roguz/freesound/freesound-perceptual_similarity/bash_scripts/"
export PATH="$SCRIPT:$PATH"

#############################################################################

essentia_extractor_model_output_to_evaluation.sh 100 "nn"
essentia_extractor_model_output_to_evaluation.sh 100 "dot"
essentia_extractor_model_output_to_evaluation.sh 200 "nn"
essentia_extractor_model_output_to_evaluation.sh -1 "nn"

#############################################################################