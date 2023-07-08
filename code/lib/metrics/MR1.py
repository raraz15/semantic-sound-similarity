""" This script contains functions to calculate R1 and related 
metrics such as Instance-based MR1, Label-based MR1 and Family-based MR1."""

import numpy as np

from ..utils import get_labels_of_fname, find_indices_containing_label, get_all_labels

####################################################################################
# R1

def R1(query_fname, result, df, query_label=None):
    """Returns the first index of a relevant result in the result list.
    If the query_label is not None, it evaluates relevance by inclusion of 
    the query_label in the result labels. If the query_label is None, it evaluates 
    relevance by intersection of the query labels and the result labels."""

    query_labels = get_labels_of_fname(query_fname, df)
    for i,ref_result in enumerate(result):
        ref_fname = ref_result["result_fname"]
        ref_labels = get_labels_of_fname(ref_fname, df)
        # Check if the ref is relevant
        if query_label is not None:
            if query_label in ref_labels:
                return i
        else:
            if len(query_labels.intersection(ref_labels)) > 0:
                return i # Return the rank of the first match

####################################################################################
# Different Mean Rank1s

def instance_based_MR1(embeddings, df):
    """Calculate the MR1 metric for the given embeddings and ground truth. 
    It treats each embedding as a query, removes the query from the database 
    and calculates the R1 for each query. Then it calculates the mean of the R1s.
    Since each instance is treated as a query, this metric is called instance-based
    MR1 and its a micro averaged metric."""

    r1s = []
    for i,(query_fname,query_embed) in enumerate(embeddings.items()):
        if (i+1)%1000==0:
            print(f"{i+1}/{len(embeddings)}")
        # Remove the query from the corpus
        _embeddings = {k: v for k, v in embeddings.items() if k != str(query_fname)}
        # Compare the query to the rest of the corpus
        distances = [[ref_name, np.linalg.norm(query_embed-ref_embed)] for ref_name, ref_embed in _embeddings.items()]
        ranked_fnames = [{'result_fname': r[0]} for r in sorted(distances, key=lambda x: x[1])]
        # Calculate the Ranking of the first element
        r1s.append(R1(query_fname, ranked_fnames, df))
    # Calculate the mean of the R1s
    mr1 = sum(r1s)/len(r1s)
    return mr1

def calculate_mr1_for_labels(embeddings, df):

    # Get all the labels from the df
    labels = get_all_labels(df)

    label_mr1s = []
    for i, query_label in enumerate(labels):
        if (i+1)%20==0:
            print(f"{i+1}/{len(labels)}")
        # Get the fnames containing this label
        fnames_with_label = df[find_indices_containing_label(query_label, df)]["fname"].to_list()
        # For each fname, get the embedding and compare it to the rest of the corpus
        r1s = []
        for query_fname in fnames_with_label:
            query_embed = embeddings[str(query_fname)]
            # Remove the query from the corpus
            _embeddings = {k: v for k, v in embeddings.items() if k != str(query_fname)}
            # Compare the query to the rest of the corpus
            distances = [[ref_name, np.linalg.norm(query_embed-ref_embed)] for ref_name, ref_embed in _embeddings.items()]
            ranked_fnames = [{'result_fname': r[0]} for r in sorted(distances, key=lambda x: x[1])]
            # Calculate the Ranking of the first element
            r1s.append(R1(query_fname, ranked_fnames, df, query_label=query_label))
        # Calculate the mean average precision@n for this label
        label_mr1 = sum(r1s)/len(r1s)
        # Append the results
        label_mr1s.append([query_label, label_mr1, len(fnames_with_label)])
    # Sort the label MR1s by the mAP@n value
    label_mr1s.sort(key=lambda x: x[1], reverse=False)
    return label_mr1s, ["label", "MR1", "n_relevant"]

def label_based_mr1(label_mr1s):
    """ Calculates the macro mean rank1 (map) for the whole dataset. 
    That is, the Mr1 values of each label is averaged."""

    return sum([label_mr1[1] for label_mr1 in label_mr1s])/len(label_mr1s)

def family_based_mr1(label_MR1s, families=dict()):
    """ Using families dict which specifies the family names and the list of all its child names
    inside the FSD50K labels, averages the MR1 for each family."""

    # Calculate the MR1 for each family
    family_maps = []
    for family_name, child_names in families.items():
        # Get the label MR1s for this family
        family_label_maps = [label_map[1] for label_map in label_MR1s if label_map[0] in child_names]
        # Calculate the family MR1
        family_mr1 = sum(family_label_maps)/len(family_label_maps)
        # Append the results
        family_maps.append([family_name, family_mr1])
    # Sort the family MR1s by the mAP@n value
    family_maps.sort(key=lambda x: x[1], reverse=False)
    return family_maps, ["family", "mr1"]
