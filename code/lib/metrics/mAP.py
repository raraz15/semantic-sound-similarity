""" This script contains functions to calculate the mean average precision (MAP) metric."""

####################################################################################

def precision_at_k(relevance, k):
    """ Calculate precision@k where k is and index in range(0,len(relevance)). Since 
    relevance is a list of 0s (fp) and 1s (tp), the precision is the sum of the 
    relevance values up to k, which is equal to sum of tps up to k, divided by the 
    length (tp+fp)."""

    assert k>=0 and k<len(relevance), "k must be an index in range(0,len(relevance))"
    assert set(relevance).issubset({0,1}), "Relevance values must be 0 or 1"

    return sum(relevance[:k+1])/(k+1)

####################################################################################
# Average Precision

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