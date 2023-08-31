#!/bin/bash

SCRIPT_DIR="$(pwd)/scripts/pipelines/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

# Define here the variables of the experiment.
variable=(80)

#############################################################################

for v in ${variable[@]}; do
  freesound-prepare_search_evaluate.sh $v
done

# Compare the results of the experiments
python code/plot_evaluation_results_comparisons.py $MODEL_NAME

#############################################################################