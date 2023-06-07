"""Script to contain the metrics used for evaluation."""

# TODO: ncdg

####################################################################################
# Utilities

def get_labels(fname, df):
    """Returns the set of labels of the fname from the dataframe."""

    return set(df[df["fname"]==int(fname)]["labels"].values[0].split(","))

def evaluate_relevance(query_fname, result, df, query_label=None):
    """ Evaluates the relevance of a result for a query. By default, A result is considered
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

####################################################################################
# mAP Related Metrics

def precision_at_k(relevance, k):
    """ Calculate precision@k where k is and index in range(0,len(relevance)). Since relevance
    is a list of 0s (fp) and 1s (tp), the precision is the sum of the relevance values up to k, 
    which is equal to sum of tps up to k, divided by the length (tp+fp)."""

    return sum(relevance[:k+1])/(k+1)

def average_precision(relevance):
    """ Calculate the average prediction for a list of relevance values. The average 
    precision is defined as the average of the precision@k values of the relevant 
    documents. If there are no relevant documents, the average precision is defined 
    to be 0. """

    assert set(relevance).issubset({0,1}), "Relevance values must be 0 or 1"

    # Number of relevant documents
    tp = sum(relevance)
    # If there are no relevant documents, define the average precision as 0
    if tp==0:
        ap = 0
    else:
        # Calculate average precision
        total = sum([rel_k*precision_at_k(relevance,k) for k,rel_k in enumerate(relevance)])
        ap = total / tp
    return ap

def test_average_precision():
    """ Test the average precision function."""

    results = [
            [[0,0,0,0,0,0], 0.0],
            [[1,1,0,0,0,0], 1.0],
            [[0,0,0,0,1,1], 0.266],
            [[0,1,0,1,0,0], 0.5],
            ]
    for result,answer in results:
        delta = average_precision(result)-answer
        if abs(delta)>0.001:
            print("Error")
            import sys
            sys.exit(1)

def calculate_micro_map_at_k(results_dict, df, k):
    """ Calculates the mean average precision@k (map@k) for the whole dataset (Micro 
    metric). That is, each element in the dataset is considered as a query and the 
    average precision@k is calculated for each query result. The mean of all these 
    values is returned."""

    # Calculate the average precision for each query
    aps = []
    for query_fname, result in results_dict.items():
        # Evaluate the relevance of the result
        relevance = evaluate_relevance(query_fname, result[:k], df) # Cutoff at k
        # Calculate the average precision with the relevance
        ap = average_precision(relevance)
        # Append the results
        aps.append(ap)
    # Mean average precision for the whole dataset
    map_at_k = sum(aps)/len(aps)
    return map_at_k

def calculate_map_at_k_for_labels(results_dict, df, k=15):
    """ For each label in the dataset, the elements containing that label are considered 
    as queries and the mean average precision@k (map@k) is calculated. Here, relevance is 
    defined as: if a result contains the query label it. We also weigh each label's map@k 
    by the number of elements containing that label divided by total number of elements."""

    # Get all the labels from the df
    labels = set([l for ls in df["labels"].apply(lambda x: x.split(",")).to_list() for l in ls])
    # Calculate map@k for each label and the weighted map@k 
    label_maps = []
    for query_label in labels:
        # Get the fnames containing this label
        fnames_with_label = df[df["labels"].apply(lambda x: query_label in x)]["fname"].to_list()
        # For each fname containing the label, calculate the total tp and fp
        label_aps = []
        for query_fname in fnames_with_label:
            # Get the result for this query
            result = results_dict[str(query_fname)][:k] # Cutoff at k
            # Evaluate the relevance of the result
            relevance = evaluate_relevance(query_fname, result, df, query_label=query_label)
            # Calculate ap@k for this query
            ap = average_precision(relevance)
            # Append the results
            label_aps.append(ap)
        # Calculate the mean average precision for this label
        label_map_at_k = sum(label_aps)/len(label_aps)
        # Calculate the weight of the label and weigh the label map@k
        label_weight = len(fnames_with_label)/len(df)
        weighted_label_map_at_k = label_map_at_k * label_weight
        # Append the results
        label_maps.append([query_label, label_map_at_k, weighted_label_map_at_k, label_weight])
    # Sort the label maps by the map@k value
    label_maps.sort(key=lambda x: x[1], reverse=True)
    return label_maps, ["label", "map@15", "weighted_map@k", "weight"]

def calculate_macro_map(label_maps):
    """ Calculates the macro mean average precision (map) for the whole dataset. 
    That is, the map@k values for each label is averaged."""

    return sum([label_map[1] for label_map in label_maps])/len(label_maps)

def calculate_weighted_macro_map(label_maps):
    """ Calculates the weighted macro mean average precision (map) for the whole
    dataset. That is, the weighted map@k values for each label is averaged."""

    return sum([label_map[2] for label_map in label_maps])/len(label_maps)

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