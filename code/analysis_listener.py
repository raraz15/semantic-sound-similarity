"""Listen to randomly selected target and query sounds from an analysis file."""

import os
import re
import json
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd

from lib.metrics import evaluate_relevance,average_precision, find_indices_containing_label, get_labels
from lib.directories import *

FREESOUND_STRING = '<iframe frameborder="0" scrolling="no" \
                    src="https://freesound.org/embed/sound/iframe/{}/simple/medium/" \
                    width="481" height="86"></iframe>'

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

def display_query_and_similar_sounds(query_fname, df, model_result_dcts, N=15, header=None, query_label=None):

    # Display the header if provided
    if header is not None:
        st.header(header)

    # Display the query sound
    with st.container():
        st.subheader("Random Query Sound")
        st.caption(f"Sound ID: {query_fname}")
        # Get the query labels and highlight the query label if provided
        query_labels = get_labels(query_fname, df)
        query_labels = [f":blue[{query_label}]" if label==query_label else label for label in query_labels]
        st.write(f"Labels: {', '.join(query_labels)}")
        # Display the query sound
        st.components.v1.html(FREESOUND_STRING.format(query_fname))
    st.divider()

    # Convert the query labels to a set for faster lookup
    query_labels = set(query_labels)

    # Display the top N similar sounds for each embedding-search combination
    st.subheader(f"Top {N} Similar Sounds for Embedding-Search Combination(s)")
    st.write("Labels that are common to the query and the reference sound are highlighted in :green[green].")
    if query_label is not None:
        st.write(f"The :blue[query label] is marked with blue.")
    st.divider()
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
                if model_name=="fs-essentia-extractor_legacy":
                    st.subheader("-")
                for v in variant.split("-"):
                    st.subheader(v)
                if model_name=="fs-essentia-extractor_legacy":
                    st.subheader("-")
                st.subheader(f"{model_result_dct['search']} Search")

                # Calculate and display the relevance and theaverage precision for the query sound with this embedding-search combination
                if query_label is None: # If a query label is provided, check for inclusion
                    relevance_intersection = evaluate_relevance(query_fname, 
                            model_result_dct['results'][query_fname][:N], 
                            df)
                    ap_at_15_intersection = average_precision(relevance_intersection)
                    st.write(f":green[{sum(relevance_intersection)}] sound(s) share a label with the query sound. "
                            f"[AP@{N}: **{ap_at_15_intersection:.3f}**]")
                else: # If a query label is provided, check for inclusion and intersection
                    # Inclusion
                    relevance_inclusion = evaluate_relevance(query_fname, 
                                                model_result_dct['results'][query_fname][:N], 
                                                df,
                                                query_label=query_label)
                    ap_at_15_inclusion = average_precision(relevance_inclusion)
                    st.write(f":blue[{sum(relevance_inclusion)}] sound(s) contain the query label. "
                             f"[AP@{N}: **{ap_at_15_inclusion:.3f}**]")
                    # Intersection
                    relevance_intersection = evaluate_relevance(query_fname, 
                                                model_result_dct['results'][query_fname][:N], 
                                                df)
                    ap_at_15_intersection = average_precision(relevance_intersection)
                    # See how many items can be potentially relevant
                    diff = [1 if (x==0 and y==1) else 0 for x,y in zip(relevance_inclusion, relevance_intersection)]
                    st.write(f":green[{sum(diff)}] sound(s) share a label with the query sound other than the query label. "
                             f"[AP@{N}: **{ap_at_15_intersection:.3f}**]")
                st.divider()

                # Display the top N similarity results
                for j,result in  enumerate(model_result_dct["results"][query_fname][:N]):
                    # Determine the score type and display it
                    if model_result_dct["search"]=="Nearest Neighbor":
                        st.write(f"Ranking: {j+1} - Distance: {result['score']:.3f}")
                    elif model_result_dct["search"]=="Dot Product":
                        st.write(f"Ranking: {j+1} - Score: {result['score']:.3f}")
                    ref_fname = result["result_fname"]
                    st.caption(f"Sound ID: {ref_fname}")
                    # Highlight the common labels between the query and the reference sound
                    ref_labels = get_labels(ref_fname,df)
                    for common_label in query_labels.intersection(set(ref_labels)):
                        ref_labels = [f":green[{label}]" if label==common_label else label for label in ref_labels]
                    # Highlight the query label if provided
                    ref_labels = [f":blue[{label}]" if label==query_label else label for label in ref_labels]
                    # Display the highlighted labels
                    st.write(f"Labels: {', '.join(ref_labels)}")
                    # Display the result sound
                    st.components.v1.html(FREESOUND_STRING.format(ref_fname))

# TODO: display the AP@N for the selected label for each embedding-search combination
def get_subsets(sound_classes, exclusion_classes, df, model_results_dcts, N=15):
    """Get the subset of sounds containing all the selected labels.
    If no sound contains all the labels, return an error."""

    # If only one label is selected, display the results for that label
    if len(sound_classes)==1:
        # Find which indices contain the selected label
        indices = find_indices_containing_label(sound_classes[0], df)
        # If there are exclusion classes, remove them from the indices
        if len(exclusion_classes)>0:
            # Find which indices contain the excluded labels
            exclusion_indices = pd.Series(dtype=bool)
            for sound_class in exclusion_classes:
                exclusion_indices = exclusion_indices | find_indices_containing_label(sound_class, df)
            # Get the subset of sounds containing the selected label but not the excluded labels
            indices = indices & ~exclusion_indices
            # If there are no sounds containing the selected label, return an error
            if sum(indices)==0:
                st.error(f"No sound contains the :blue[{sound_classes[0]}] label "
                        f"but not the :red[{', '.join(exclusion_classes)}] label(s). Choose Again.")
                return
        # Get the filenames of the subset
        fnames_of_intersection = df[indices]["fname"].to_list()
        # Get a random sound from the subset
        fname = str(random.choice(fnames_of_intersection))
        # Header to display above the results
        header = f"There are {len(fnames_of_intersection)} sounds containing the :blue[{', '.join(sound_classes)}] label"
        if len(exclusion_classes)>0:
            header += f" but not the :red[{', '.join(exclusion_classes)}] label(s)."
        else:
            header += "."
        # Display the results
        display_query_and_similar_sounds(fname, 
                                        df, 
                                        model_results_dcts, 
                                        N=N, 
                                        header=header,
                                        query_label=sound_classes[0])
    else:
        # Get the subset of sounds containing all the selected labels
        indices = pd.Series(dtype=bool)
        for sound_class in sound_classes:
            indices = indices & find_indices_containing_label(sound_class, df)
        # If there are exclusion classes, remove them from the indices
        if len(exclusion_classes)>0:
            # Find which indices contain the excluded labels
            exclusion_indices = pd.Series(dtype=bool)
            for sound_class in exclusion_classes:
                exclusion_indices = exclusion_indices | find_indices_containing_label(sound_class, df)
            # Get the subset of sounds containing the selected label but not the excluded labels
            indices = indices & ~exclusion_indices
            # If there are no sounds containing the selected label, return an error
            if sum(indices)==0:
                st.error(f"No sound contains the :blue[{sound_classes[0]}] label "
                        f"but not the :red[{', '.join(exclusion_classes)}] label(s). Choose Again.")
                return
        fnames_of_intersection = df[indices]["fname"].to_list()
        # If no sound contains all the labels, return an error
        if fnames_of_intersection==[]:
            st.error("No sound found containing all the selected labels. Choose Again.")
            return
        else:
            # Get a random sound from the subset
            fname = str(random.choice(fnames_of_intersection))
            # Header to display above the results
            header = f"There are {len(fnames_of_intersection)} sounds containing *{', '.join(sound_classes)}* labels."
            # Display the results
            display_query_and_similar_sounds(fname, 
                                            df, 
                                            model_results_dcts, 
                                            N=N, 
                                            header=header,
                                            query_label=None)

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
                                   options=all_labels,
                                   default=[],
                                   help="Select the sound categories you want to search for.")
    exclusion_classes = st.multiselect("Any Classes you want to Exclude?",
                                        options=all_labels,
                                        default=[],
                                        help="If you select a class here, "
                                              "no sound containing that class will be returned.",
                                              )
    # Display the results with the selected categories
    st.button(label=":speaker:", 
            on_click=get_subsets, 
            args=(sound_classes, exclusion_classes, df, model_results_dcts), 
            kwargs={"N":args.N}
            )