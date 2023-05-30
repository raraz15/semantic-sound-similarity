"""Listen to randomly selected target and query sounds from an analysis file."""

import os
import re
import sys
import json
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd

from evaluate import calculate_average_precision
from directories import *

FREESOUND_STRING = '<iframe frameborder="0" scrolling="no" \
                    src="https://freesound.org/embed/sound/iframe/{}/simple/medium/" \
                    width="481" height="86"></iframe>'

st.set_page_config(page_title="Sound Similarity", page_icon=":loud_sound:", layout="wide")

# TODO: allow mutations?
@st.cache_resource()
def load_gt():
    """Load the ground truth file and return the dataframe and all labels."""
    df = pd.read_csv(GT_PATH)
    df.labels = df.labels.apply(lambda x: re.sub("_", " ", x))
    all_labels = sorted(list(set([y for x in df.labels.to_list() for y in x.split(",")])))
    return df, all_labels

@st.cache_resource
def load_results(paths):
    # For each analysis file, load the results
    model_results_dcts = []
    for path in paths:
        # Load the analysis file
        similarity_dict = {}
        with open(path ,"r") as infile:
            for jline in infile:
                result_dict = json.loads(jline)
                similarity_dict[result_dict["query_fname"]] = result_dict["results"]
        # Get the search type
        if result_dict["search"]=="nearest_neighbour":
            search = "Nearest Neighbor"
        elif result_dict["search"]=="dot":
            search = "Dot Product"
        else:
            raise ValueError("Unknown search type.")
        # Get the embeddings name
        embeddings_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        # Append the results
        model_results_dcts.append({
                                "embeddings_name": embeddings_name, 
                                "search": search, 
                                "results": similarity_dict, 
                                "fnames": list(similarity_dict.keys()), 
                                })
    return model_results_dcts

def display_query_and_similar_sound(query_fname, df, model_result_dcts, N=15, header=None):

    # Get the labels for the query sound
    query_labels = df[df.fname==int(query_fname)].labels.values[0].replace(",", ", ")
    if header is not None:
        st.header(header)
    with st.container():
        st.subheader("Query Sound")
        st.caption(f"Sound ID: {query_fname}")
        st.caption(f"Labels: {query_labels}")
        st.components.v1.html(FREESOUND_STRING.format(query_fname))
    st.divider()
    st.subheader(f"Top {N} Similar Sounds for each Embedding")
    with st.container():
        # Create a column for each embedding-search combination
        columns = st.columns(len(model_result_dcts))
        for i, model_result_dct in enumerate(model_result_dcts):

            # Get the model name and variant
            if "Agg" not in model_result_dct['embeddings_name']:
                model_name, variant = model_result_dct['embeddings_name'].split("-PCA")
                variant = "PCA"+"".join(variant)
            else:
                model_name, variant = model_result_dct['embeddings_name'].split("-Agg")
                variant = "Agg"+"".join(variant)
            # Calculate the average precision for the query sound with this embedding
            ap = calculate_average_precision(query_fname, model_result_dct['results'][query_fname], df)

            # Display the results for this embedding-search combination
            with columns[i]:
                # Display the model name, variant and search
                st.subheader(model_name)
                for v in variant.split("-"):
                    st.subheader(v)
                st.subheader(f"{model_result_dct['search']} Search")
                # Display the average precision
                st.write(f"Average Precision@15 for this result is: {ap:.3f}")
                # Display the results
                for j,result in  enumerate(model_result_dct["results"][query_fname][:N]):
                    ref_fname = result["result_fname"]
                    ref_labels = df[df.fname==int(ref_fname)].labels.values[0].replace(",", ", ")
                    st.write(f"Ranking: {j+1} - Score: {result['score']:.3f}")
                    st.caption(f"Sound ID: {ref_fname}")
                    st.caption(f"Labels: {ref_labels}")
                    st.components.v1.html(FREESOUND_STRING.format(ref_fname))

def get_subsets(sound_classes, df, model_results_dcts, N=15):

    # Get the subset of sounds containing all the selected labels
    indices = df.labels.str.contains(sound_classes[0])
    if len(sound_classes)>1:
        for sound_class in sound_classes[1:]:
            indices = indices & df.labels.str.contains(sound_class)
    fnames_of_class = df[indices].fname.to_list()
    # If no sound contains all the labels, return an error
    if fnames_of_class==[]:
        st.error("No sound found containing all the selected labels. Choose Again.")
        return
    else:
        # Get a random sound from the subset
        fname = str(random.choice(fnames_of_class))
        # Header to display above the results
        header = f"Random Sound Containing '{', '.join(sound_classes)}' Label(s)"
        # Display the results
        display_query_and_similar_sound(fname, 
                                        df, 
                                        model_results_dcts, 
                                        N=N, 
                                        header=header)

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p0", "--path0", type=str, default=None, 
                        help='Similarity Result Path 0.')
    parser.add_argument("-p1", "--path1", type=str, default=None, 
                        help='Similarity Result Path 1.')
    parser.add_argument("-p2", "--path2", type=str, default=None, 
                        help='Similarity Result Path 2.')
    parser.add_argument("-p3", "--path3", type=str, default=None, 
                        help='Similarity Result Path 3.')
    parser.add_argument("-p4", "--path4", type=str, default=None, 
                        help='Similarity Result Path 4.')
    parser.add_argument('-N', type=int, default=15, 
                        help="Number of top entries to display.")
    args=parser.parse_args()

    # Get the none paths
    paths = [path for path in [args.path0, args.path1, args.path2, args.path3, args.path4] if path is not None]

    # Load the ground truth and results
    df, all_labels = load_gt()
    model_results_dcts = load_results(paths)

    # Display general information
    st.title("Evaluate Sound Similarity Results for Embeddings")
    st.header("Choose Sound Categories and Click The Speaker Icon.")
    st.write("This will select a random sound containing all the categories and display \
             the top similar sounds returned by each embedding.")

    # Select sound categories
    sound_classes = st.multiselect("FSD50K Sound Classes", 
                                   options=all_labels)
    # Display the results with the selected categories
    st.button(label=":speaker:", 
            on_click=get_subsets, 
            args=(sound_classes, df, model_results_dcts), 
            kwargs={"N":args.N}
            )