#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/bash_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

yamnet_search_and_evaluate.sh "Agg_mean-PCA_100-Norm_False" "dot"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_100-Norm_True" "dot"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_200-Norm_False" "dot"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_200-Norm_True" "dot"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_1024-Norm_False" "dot"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_1024-Norm_True" "dot"

yamnet_search_and_evaluate.sh "Agg_mean-PCA_100-Norm_False" "nn"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_100-Norm_True" "nn"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_200-Norm_False" "nn"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_200-Norm_True" "nn"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_1024-Norm_False" "nn"
yamnet_search_and_evaluate.sh "Agg_mean-PCA_1024-Norm_True" "nn"