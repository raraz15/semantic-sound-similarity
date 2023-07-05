"""Script to contain the metrics used for evaluation."""

import re

# TODO: ncdg

####################################################################################
# Utilities

def get_labels(fname, df):
    """Returns the set of labels of the fname from the dataframe."""

    return set(df[df["fname"]==int(fname)]["labels"].values[0].split(","))

def find_indices_containing_label(label, df):
    """Returns the list of fnames that contain the label. Uses a regular expression to
    find the label in the labels column."""

    label = re.escape(label)
    # Create the pattern to search the label in the labels column
    pattern = ""
    for p in ["\A"+label+",|", ","+label+",|", ","+label+"\Z", r"|\A"+label+r"\Z"]:
        pattern += p
    pattern = re.compile(r"(?:"+pattern+r")") # Compile the pattern with matching group
    return df["labels"].str.contains(pattern)

def evaluate_relevance(query_fname, result, df, query_label=None):
    """Evaluates the relevance of a result for a query. By default, A result is considered
    relevant if it has at least one label in common with the query. If a query label is 
    provided, a result is considered relevant if it contains the label. Relevance: list of 
    relevance values (1: relevant, 0: not relevant)."""

    # Get the labels of the query
    query_item_labels = get_labels(query_fname, df)
    # Evaluate the relevance of each retrieved document
    relevance = []
    for ref_result in result:
        ref_fname = ref_result["result_fname"]
        ref_item_labels = get_labels(ref_fname, df)
        if query_label is None:
            # Find if the retrieved element is relevant
            if len(query_item_labels.intersection(ref_item_labels)) > 0:
                relevance.append(1)
            else:
                relevance.append(0)
        else:
            # Find if the retrieved element contains the label
            if query_label in ref_item_labels:
                relevance.append(1)
            else:
                relevance.append(0)
    return relevance

def precision_at_k(relevance, k):
    """ Calculate precision@k where k is and index in range(0,len(relevance)). Since 
    relevance is a list of 0s (fp) and 1s (tp), the precision is the sum of the 
    relevance values up to k, which is equal to sum of tps up to k, divided by the 
    length (tp+fp)."""

    assert set(relevance).issubset({0,1}), "Relevance values must be 0 or 1"

    return sum(relevance[:k+1])/(k+1)

####################################################################################
# Average Precision and Average Precision@n

def average_precision(relevance, n_relevant):
    """ Calculate the average presicion for a list of relevance values. The average 
    precision is defined as the 'average of the precision@k values of all the relevant 
    documents in the collection'. If there are no relevant documents, the average 
    precision is defined to be 0. This calculation is based on the definition 
    in https://link.springer.com/referenceworkentry/10.1007/978-1-4899-7993-3_482-2
    """

    assert sum(relevance)==n_relevant, "Number of relevant documents does not match relevance list"

    # If there are no relevant documents, define the average precision as 0
    if n_relevant==0:
        ap = 0
    else:
        total = sum([rel_k*precision_at_k(relevance,k) for k,rel_k in enumerate(relevance)])
        ap = total / n_relevant
    return ap

def average_precision_at_n(relevance, n, n_relevant=None):
    """ Calculate the average presicion@n for a list of relevance values. The average 
    precision@n is defined as the 'average of the precision@k values of the relevant 
    documents at the top n rankings'. If there are no relevant documents, the average 
    precision is defined to be 0. This calculation is based on the definition 
    in https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_487
    You can provide n_relevant if you know the number of relevant documents in the
    collection and it is smaller than n. This way you do not punish rankings for 
    documents with less than n relevant documents in the collection.
    """

    assert n>0, "n must be greater than 0"
    assert len(relevance)==n, f"Number of relevance values={len(relevance)} does not match n={n}"

    # If there are no relevant documents in top n, define the average precision@n as 0
    if sum(relevance)==0:
        ap_at_n = 0
    else:
        total = sum([rel_k*precision_at_k(relevance,k) for k,rel_k in enumerate(relevance)])
        # If n_relevant is provided, compare it with ranking length and
        # use the smaller to normalize the total
        normalization = min(n,n_relevant) if n_relevant is not None else n
        ap_at_n = total / normalization
    return ap_at_n

def test_average_precision_at_n():
    """ Test the average_precision_at_n function."""

    tests = [
            [[0,0,0,0,0,0], 6, 10, 0.0],
            [[1,1,0,0,0,0], 6,  2, 1.0],
            [[0,0,0,0,1,1], 6,  2, 0.266],
            [[0,1,0,1,0,0], 6,  2, 0.5],
            [[1,0,0,1,0,1], 6,  3, 0.666],
            [[1,1,1,0,0], 5,  8, 0.6]
            ]
    for relevance,length,n_relevant,answer in tests:
        delta = average_precision_at_n(relevance, length, n_relevant)-answer
        if abs(delta)>0.001:
            print("Error at test_average_precision_at_n")
            import sys
            sys.exit(1)

####################################################################################
# AP@n Related Metrics

def instance_based_map_at_n(results_dict, df, n, n_relevant=None):
    """ Calculates the mean of the average precision@n (mAP@n) over the whole dataset. 
    That is, each element in the dataset is considered as a query and the average 
    precision@n (ap@n) is calculated for the ranking. The mean of all these values 
    is returned (Micro metric)."""

    # Calculate the average precision for each query
    aps = []
    for query_fname, result in results_dict.items():
        # Evaluate the relevance of the result
        relevance = evaluate_relevance(query_fname, result[:n], df) # Cutoff at n
        # Calculate the average precision with the relevance
        ap_at_n = average_precision_at_n(relevance, n, n_relevant=n_relevant)
        # Append the results
        aps.append(ap_at_n)
    # Mean average precision for the whole dataset
    map_at_k = sum(aps)/len(aps)
    return map_at_k

def calculate_map_at_n_for_labels(results_dict, df, n):
    """ For each label in the dataset, the elements containing that label are considered 
    as queries and the average precision@n (AP@n) is averaged for all the rankings. Here, 
    relevance is defined as: if a result contains the query label. """

    # Get all the labels from the df
    labels = set([l for ls in df["labels"].apply(lambda x: x.split(",")).to_list() for l in ls])
    # Calculate map@k for each label
    label_maps = []
    for query_label in labels:
        # Get the fnames containing this label
        fnames_with_label = df[find_indices_containing_label(query_label, df)]["fname"].to_list()
        # Find how many elements contain this label, for the case of FSD50K.eval, 
        # we know that n_relevant is always bigger than 15
        n_relevant = len(fnames_with_label)
        # For each fname containing the label, aggregate the AP@n
        label_aps = []
        for query_fname in fnames_with_label:
            # Get the result for this query
            result = results_dict[str(query_fname)][:n] # Cutoff at n
            # Evaluate the relevance of the result
            relevance = evaluate_relevance(query_fname, result, df, query_label=query_label)
            # Calculate ap@n for this query
            ap_at_n = average_precision_at_n(relevance, n, n_relevant=n_relevant)
            # Append the results
            label_aps.append(ap_at_n)
        # Calculate the mean average precision@n for this label
        label_map_at_n = sum(label_aps)/len(label_aps)
        # Append the results
        label_maps.append([query_label, label_map_at_n, n_relevant])
    # Sort the label maps by the mAP@n value
    label_maps.sort(key=lambda x: x[1], reverse=True)
    return label_maps, ["label", f"map@{n}", "n_relevant"]

def label_based_map_at_n(label_maps):
    """ Calculates the macro mean average precision (map) for the whole dataset. 
    That is, the map@k values for each label is averaged."""

    return sum([label_map[1] for label_map in label_maps])/len(label_maps)

def family_based_map_at_n(label_maps, families=dict()):
    """ Using families dict which specifies the family names and the list of all its child names
    inside the FSD50K labels, averages the map@n for each family."""

    # Calculate the map@k for each family
    family_maps = []
    for family_name, child_names in families.items():
        # Get the label maps for this family
        family_label_maps = [label_map[1] for label_map in label_maps if label_map[0] in child_names]
        # Calculate the family map@k
        family_map_at_n = sum(family_label_maps)/len(family_label_maps)
        # Append the results
        family_maps.append([family_name, family_map_at_n])
    # Sort the family maps by the mAP@n value
    family_maps.sort(key=lambda x: x[1], reverse=True)
    return family_maps, ["family", "map"]

####################################################################################
# Ranking Related Metrics

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