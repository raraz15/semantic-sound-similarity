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

#############################################################################