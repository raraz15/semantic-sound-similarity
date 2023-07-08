import numpy as np

from .utils import get_fname

def dot_product_search(query, corpus, N):
    """Computes pairwise dot product similarities and returns the indices of top N. 
    Assumes that the query is aggregated and the query is removed from the corpus. 
    Returns a dictionary with the query fname and a list of dictionaries with 
    the results and their scores."""

    # Get the query embedding and path
    query_embed, query_path = query[0], query[1]
    assert len(query_embed.shape)==1, "To use dot product search, "\
        f"queries should be aggregated! {query_embed.shape}"
    # If N is -1, return all the results
    if N==-1:
        N = len(corpus)
    # For each reference in the dataset, compute the dot product with the query
    products = [np.dot(query_embed, ref[0]) for ref in corpus]
    # Get the indices of the top N similar elements in the corpus
    indices = np.argsort(products)[::-1][:N]
    # Return the results
    return {"query_fname": get_fname(query_path), 
            "results": [{"result_fname": get_fname(corpus[i][1]), 
                         "score": products[i]} for i in indices],
            "search": "dot_product"
            }

def nn_search(query, corpus, N):
    """Computes pairwise distances and returns the indices of bottom N. 
    Assumes that the query is aggregated and the query is removed from the corpus. 
    Returns a dictionary with the query fname and a list of dictionaries with 
    the results and their scores."""

    # If N is -1, return all the results
    if N==-1:
        N = len(corpus)

    query_embed, query_path = query[0], query[1]
    # For each reference in the dataset, compute the distance to the query
    distances = [np.linalg.norm(query_embed-ref[0]) for ref in corpus]
    # Get the indices of the top N closest elements in the corpus
    indices = np.argsort(distances)[:N]
    # Return the results
    return {"query_fname": get_fname(query_path), 
            "results": [{"result_fname": get_fname(corpus[i][1]), 
                         "score": distances[i]} for i in indices],
            "search": "nearest_neighbour"
            }

def search_similar_sounds(query, corpus, N, algo="dot"):
    if algo=="dot":
        return dot_product_search(query, corpus, N)
    elif algo=="nn":
        return nn_search(query, corpus, N)
    else:
        raise NotImplementedError