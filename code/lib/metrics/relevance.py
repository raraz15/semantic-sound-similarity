from ..utils import get_labels_of_fname

####################################################################################
# Utilities

def evaluate_relevance(query_fname, result, df, query_label=None, cutoff=False):
    """Evaluates the relevance of a result for a query. By default, A result is considered
    relevant if it has at least one label in common with the query. If a query label is 
    provided, a result is considered relevant if it contains the label. You can cutoff the
    relevance from the last 1 by setting cutoff to True.
    Returns:
        Relevance: list of relevance values (1: relevant, 0: not relevant)."""

    # Get the labels of the query
    query_item_labels = get_labels_of_fname(query_fname, df)
    # Evaluate the relevance of each retrieved document
    relevance = []
    for ref_result in result:
        ref_fname = ref_result["result_fname"]
        ref_item_labels = get_labels_of_fname(ref_fname, df)
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
    # Cutoff the relevance from the last 1 if cutoff is True
    if cutoff:
        last_one = relevance[::-1].index(1)
        relevance = relevance[:-last_one]
    return relevance
