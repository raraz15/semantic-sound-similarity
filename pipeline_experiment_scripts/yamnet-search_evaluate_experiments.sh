#!/bin/bash

SCRIPT_DIR="/home/roguz/freesound/freesound-perceptual_similarity/pipeline_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

yamnet-search_evaluate.sh "Agg_mean-PCA_100-Norm_False" "dot"
yamnet-search_evaluate.sh "Agg_mean-PCA_100-Norm_True" "dot"
yamnet-search_evaluate.sh "Agg_mean-PCA_200-Norm_False" "dot"
yamnet-search_evaluate.sh "Agg_mean-PCA_200-Norm_True" "dot"
yamnet-search_evaluate.sh "Agg_mean-PCA_1024-Norm_False" "dot"
yamnet-search_evaluate.sh "Agg_mean-PCA_1024-Norm_True" "dot"

yamnet-search_evaluate.sh "Agg_mean-PCA_100-Norm_False" "nn"
yamnet-search_evaluate.sh "Agg_mean-PCA_100-Norm_True" "nn"
yamnet-search_evaluate.sh "Agg_mean-PCA_200-Norm_False" "nn"
yamnet-search_evaluate.sh "Agg_mean-PCA_200-Norm_True" "nn"
yamnet-search_evaluate.sh "Agg_mean-PCA_1024-Norm_False" "nn"
yamnet-search_evaluate.sh "Agg_mean-PCA_1024-Norm_True" "nn"