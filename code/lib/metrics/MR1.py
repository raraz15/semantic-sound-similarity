"""This module contains functions to calculate the MR1 metric."""

from .utils import get_labels

####################################################################################
# MR1

# TODO: MR1 NaNs
def R1(query_fname, result, df):
    query_labels = get_labels(query_fname, df)
    for i,ref_result in enumerate(result):
        ref_fname = ref_result["result_fname"]
        ref_labels = get_labels(ref_fname, df)
        # Find if there is a match in the labels
        if len(query_labels.intersection(ref_labels)) > 0:
            return i # Return the rank of the first match
    return None # No match

def calculate_MR1(results_dict, df):
    # Calculate the R1 for each query
    r1s = [R1(query_fname, result, df) for query_fname, result in results_dict.items()]
    # Remove entries with no matches
    r1s = [x for x in r1s if x]
    # Calculate the mean
    mr1 = sum(r1s)/len(r1s)
    return mr1