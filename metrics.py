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
# Precision with Ranking Related Metrics

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

def calculate_map_at_k(results_dict, df, k):
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
        aps.append(ap)
    # Mean average precision for the whole dataset
    map_at_k = sum(aps)/len(aps)
    return map_at_k

####################################################################################
# Precision without Ranking Related Metrics

# TODO: is this the correct way of doing?
# TODO: is applying the cutoff at k correct?
def evaluate_label_positive_rates(results_dict, df, k=15):
    """ For each label in the dataset, the elements containing that label are considered as 
    queries and the true positive and false positive rates are calculated for the result set. 
    If a result contains the query label it is considered as relevant. """

    # Get all the labels from the df
    labels = set([l for ls in df["labels"].apply(lambda x: x.split(",")).to_list() for l in ls])
    # Calculate true positives and false positives for each label
    label_positive_rates = []
    for label in labels:
        # Get the fnames containing this label
        fnames_with_label = df[df["labels"].apply(lambda x: label in x)]["fname"].to_list()
        # For each fname containing the label, calculate the total tp and fp
        tps, fps = [], []
        for query_fname in fnames_with_label:
            result = results_dict[str(query_fname)][:k] # Cutoff at k
            # Evaluate the relevance of the result
            relevance = evaluate_relevance(query_fname, result, df, label=label)
            # Calculate the true positive and false positive rates from the relevance
            tp = sum(relevance)
            fp = len(relevance)-tp
            tps.append(tp)
            fps.append(fp)
        # Append the tp and fp rates
        label_positive_rates.append([sum(tps), sum(fps), label])
    return label_positive_rates

def calculate_micro_averaged_precision(label_positive_rates):
    """ Calculate the micro-averaged precision. That is, the sum of all true positives 
    divided by the sum of all true positives and false positives."""

    return sum([tp for tp,_ in label_positive_rates]) / (sum([tp+fp for tp,fp in label_positive_rates]))

def calculate_macro_averaged_precision(label_positive_rates):
    """ Calculate the macro-averaged precision. That is, the average of the precision 
    for each label."""

    return sum([tp/(tp+fp) for tp,fp in label_positive_rates]) / len(label_positive_rates)

# TODO: is this correct?
def calculate_weighted_macro_averaged_precision(label_positive_rates):
    """ Calculate the weighted macro-averaged precision. That is, the average of the precision
    for each label weighted by the relative support of each label. The support of a label 
    is the number of elements containing that label."""

    total_support = sum([tp+fp for tp,fp in label_positive_rates])
    return sum([(tp/(tp+fp))*(tp/total_support) for tp,fp in label_positive_rates])

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