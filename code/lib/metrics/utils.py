import re

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
