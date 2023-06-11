#!/bin/bash

SCRIPT_DIR="$(pwd)/pipeline_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

# Define here the variables of the experiment.
variable=(100 200 -1)

#############################################################################

for v in ${variable[@]}; do
  essentia-prepare_search_evaluate.sh $v
done

# Compare the results of the experiments
python plot_evaluation_results.py --model=$MODEL_NAME

#############################################################################