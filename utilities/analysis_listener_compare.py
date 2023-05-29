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

from directories import *

SAMPLE_RATE = 16000

FREESOUND_STRING = '<iframe frameborder="0" scrolling="no" \
                    src="https://freesound.org/embed/sound/iframe/{}/simple/medium/" \
                    width="481" height="86"></iframe>'

st.set_page_config(page_title="Sound Similarity", page_icon=":loud_sound:", layout="wide")

def display_query_and_similar_sound(fname, df, results, N=15, header=None):

    labels =  df[df.fname==int(fname)].labels.values[0]
    if header is not None:
        st.header(header)
    with st.container():
        st.subheader("Query Sound")
        st.caption(f"Sound ID: {fname}")
        st.caption(f"Labels: {labels}")
        st.components.v1.html(FREESOUND_STRING.format(fname))
    st.divider()
    st.subheader(f"Top {N} Similar Sounds for each Embedding")
    with st.container():
        columns = st.columns(len(results))
        for i, model in enumerate(results):
            if "Agg" not in model['embeddings']:
                model_name, variant = model['embeddings'].split("-PCA")
                variant = "PCA"+"".join(variant)
            else:
                model_name, variant = model['embeddings'].split("-Agg")
                variant = "Agg"+"".join(variant)
            with columns[i]:
                st.subheader(model_name)
                for v in variant.split("-"):
                    st.subheader(v)
                st.subheader(f"{model['search']} Search")
            for j,result in  enumerate(model["results"][fname][:N]):
                fname = list(result.keys())[0]
                labels = df[df.fname==int(fname)].labels.values[0]
                with columns[i]:
                    st.write(f"Ranking: {j+1} - Score: {list(result.values())[0]:.3f}")
                    st.caption(f"Labels: {labels}")
                    st.components.v1.html(FREESOUND_STRING.format(fname))

def get_subsets(sound_classes, df, results, N=15):

    # Get the subset of sounds containing all the selected labels
    indices = df.labels.str.contains(sound_classes[0])
    if len(sound_classes)>1:
        for sound_class in sound_classes[1:]:
            indices = indices & df.labels.str.contains(sound_class)
    fnames_of_class = df[indices].fname.to_list()
    if fnames_of_class==[]:
        st.error("No sound found containing all the selected labels. Choose Again.")
        return
    idx = random.randint(0,len(fnames_of_class))
    fname = str(fnames_of_class[idx])
    header = f"Random Sound Containing '{', '.join(sound_classes)}' Label(s)"
    display_query_and_similar_sound(fname, df, results, N=N, header=header)

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

    paths = [path for path in [args.path0, args.path1, args.path2, args.path3, args.path4] if path is not None]

    # Read the meta data
    df = pd.read_csv(GT_PATH)
    df.labels = df.labels.apply(lambda x: re.sub("_", " ", re.sub(",", ", ", x)))
    all_labels = sorted(list(set([y for x in df.labels.to_list() for y in x.split(", ")])))

    results = []
    for path in paths:
        search = os.path.basename(os.path.dirname(path))
        if search=="nn":
            search = "Nearest Neighbor"
        elif search=="dot":
            search = "Dot Product"
        embeddings = os.path.basename(os.path.dirname(os.path.dirname(path)))
        # Load the analysis file
        with open(path ,"r") as infile:
            similarity_dict = json.load(infile)
        results.append({"embeddings":embeddings, 
                        "search":search, 
                        "results":similarity_dict,
                        "fnames":list(similarity_dict.keys())
                        })
    st.title("Evaluate Sound Similarity Results for Embeddings")
    st.header("Choose Sound Categories and Click The Speaker Icon.")
    st.write("This will select a random sound containing all the categories and display the top similar sounds returned by each embedding.")
    sound_classes = st.multiselect("AudioSet Top200 Sound Classes", options=all_labels)
    st.button(label=":speaker:", 
            on_click=get_subsets, 
            args=(sound_classes, df, results), 
            kwargs={"N":args.N}
            )