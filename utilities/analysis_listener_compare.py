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

from metrics import evaluate_relevance,average_precision, find_indices_containing_label
from directories import *

FREESOUND_STRING = '<iframe frameborder="0" scrolling="no" \
                    src="https://freesound.org/embed/sound/iframe/{}/simple/small/" \
                    width="375" height="30"></iframe>'

# Set the page title and icon
st.set_page_config(page_title="Sound Similarity", page_icon=":loud_sound:", layout="wide")

@st.cache_data(ttl=1800, show_spinner=False)
def load_gt():
    """Load the ground truth file and return the dataframe and all labels."""
    df = pd.read_csv(GT_PATH)
    df.labels = df.labels.apply(lambda x: re.sub("_", " ", x))
    all_labels = sorted(list(set([y for x in df.labels.to_list() for y in x.split(",")])))
    return df, all_labels

@st.cache_data(ttl=1800, show_spinner=False)
def load_results(paths):
    """For each embedding analysis, load the results and store them in a dictionary. 
    Return a list of dictionaries, one for each embedding analysis."""

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
        elif result_dict["search"]=="dot_product":
            search = "Dot Product"
        else:
            raise ValueError(f"Unknown search type. {result_dict['search']}")
        # Get the embeddings name
        embeddings_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        # Append the results
        model_results_dcts.append({
                                "embeddings_name": embeddings_name, 
                                "search": search, 
                                "results": similarity_dict, 
                                })
    return model_results_dcts

def display_query_and_similar_sound(query_fname, df, model_result_dcts, N=15, header=None, query_label=None):

    # Display the header if provided
    if header is not None:
        st.header(header)

    # Display the query sound
    query_labels = df[df.fname==int(query_fname)].labels.values[0] # Get the labels for the query sound
    with st.container():
        st.subheader("Random Query Sound")
        st.caption(f"Sound ID: {query_fname}")
        st.write(f"Labels: {query_labels.replace(',', ', ')}")
        st.components.v1.html(FREESOUND_STRING.format(query_fname))
    st.divider()

    # Split the labels into a set for easier comparison
    query_labels = set(query_labels.split(","))

    # Display the top N similar sounds for each embedding-search combination
    st.subheader(f"Top {N} Similar Sounds for each Embedding-Search Combination")
    st.write("Labels that are common to the query and the reference sound are highlighted in :green[green].")
    if query_label is not None:
        st.write(f"The :blue[query label] is marked with blue.")
    with st.container():

        # Create a column for each embedding-search combination
        columns = st.columns(len(model_result_dcts))

        # Fill columns for each embedding-search combination
        for i, model_result_dct in enumerate(model_result_dcts):

            # Display the results for this embedding-search combination
            with columns[i]:

                # Get the model name and variant and dislay it
                if "Agg" not in model_result_dct['embeddings_name']:
                    model_name, variant = model_result_dct['embeddings_name'].split("-PCA")
                    variant = "PCA"+"".join(variant)
                else:
                    model_name, variant = model_result_dct['embeddings_name'].split("-Agg")
                    variant = "Agg"+"".join(variant)
                st.subheader(model_name)
                for v in variant.split("-"):
                    st.subheader(v)
                st.subheader(f"{model_result_dct['search']} Search")

                # Calculate and display the average precision for the query sound with this embedding
                if query_label is not None:
                    relevance = evaluate_relevance(query_fname, 
                                                model_result_dct['results'][query_fname][:N], 
                                                df,
                                                query_label=query_label)
                else:
                    relevance = evaluate_relevance(query_fname, 
                                                model_result_dct['results'][query_fname][:N], 
                                                df)
                ap_at_15 = average_precision(relevance)
                st.write(f"Average Precision@{N} for this result is: {ap_at_15:.3f}")
                st.divider()

                # Display the similarity results
                for j,result in  enumerate(model_result_dct["results"][query_fname][:N]):
                    if model_result_dct["search"]=="Nearest Neighbor":
                        st.write(f"Ranking: {j+1} - Distance: {result['score']:.3f}")
                    elif model_result_dct["search"]=="Dot Product":
                        st.write(f"Ranking: {j+1} - Score: {result['score']:.3f}")
                    ref_fname = result["result_fname"]
                    st.caption(f"Sound ID: {ref_fname}")
                    # Highlight the common labels between the query and the reference sound
                    ref_labels = df[df.fname==int(ref_fname)].labels.values[0].split(",")
                    for common_label in query_labels.intersection(set(ref_labels)):
                        if common_label==query_label:
                            ref_labels = [f":blue[{common_label}]" if label==common_label else label for label in ref_labels]
                        else:
                            ref_labels = [f":green[{common_label}]" if label==common_label else label for label in ref_labels]
                    st.write(f"Labels: {', '.join(ref_labels)}")
                    # Display the result sound
                    st.components.v1.html(FREESOUND_STRING.format(ref_fname))

def get_subsets(sound_classes, df, model_results_dcts, N=15):
    """Get the subset of sounds containing all the selected labels.
    If no sound contains all the labels, return an error."""

    # Get the subset of sounds containing all the selected labels
    indices = find_indices_containing_label(sound_classes[0], df)
    if len(sound_classes)>1:
        for sound_class in sound_classes[1:]:
            indices = indices & find_indices_containing_label(sound_class, df)
        fnames_of_intersection = df[indices]["fname"].to_list()
        # If no sound contains all the labels, return an error
        if fnames_of_intersection==[]:
            st.error("No sound found containing all the selected labels. Choose Again.")
            return
        else:
            # Get a random sound from the subset
            fname = str(random.choice(fnames_of_intersection))
            # Header to display above the results
            header = f"There are {len(fnames_of_intersection)} Sounds Containing *{', '.join(sound_classes)}* Labels."
            # Display the results
            display_query_and_similar_sound(fname, 
                                            df, 
                                            model_results_dcts, 
                                            N=N, 
                                            header=header)
    else:
        fnames_of_intersection = df[indices]["fname"].to_list()
        # Get a random sound from the subset
        fname = str(random.choice(fnames_of_intersection))
        # Header to display above the results
        header = f"There are {len(fnames_of_intersection)} Sounds Containing the :blue[{', '.join(sound_classes)}] Label"
        # Display the results
        display_query_and_similar_sound(fname, 
                                        df, 
                                        model_results_dcts, 
                                        N=N, 
                                        header=header,
                                        query_label=sound_classes[0])

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