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

from essentia.standard import MonoLoader

import pandas as pd

from directories import *

SAMPLE_RATE = 16000

st.set_page_config(page_title="Sound Similarity", page_icon=":loud_sound:", layout="wide")

def display_query_and_similar_sound(fname, df, results, N=15, header=None):

    query_path = os.path.join(AUDIO_DIR, f"{fname}.wav")
    labels =  df[df.fname==int(fname)].labels.values[0]
    if header is not None:
        st.header(header)
    with st.container():
        st.subheader("Query Sound")
        st.caption(f"Sound ID: {fname}")
        st.caption(f"Labels: {labels}")
        audio = MonoLoader(filename=query_path,
                            sampleRate=SAMPLE_RATE)()
        st.audio(audio, sample_rate=SAMPLE_RATE)
    st.divider()
    st.subheader(f"Top {N} Similar Sounds")
    with st.container():
        columns = st.columns(len(results))
        for i, model in enumerate(results):
            with columns[i]:
                st.subheader(f"{model['embeddings']} - {model['search']}")
            for j,result in  enumerate(model["results"][fname][:N]):
                fname = list(result.keys())[0]
                labels = df[df.fname==int(fname)].labels.values[0]
                audio_path = os.path.join(AUDIO_DIR, f"{fname}.wav")
                audio = MonoLoader(filename=audio_path,
                                    sampleRate=SAMPLE_RATE)()
                with columns[i]:
                    st.write(f"Ranking: {j+1}")
                    st.caption(f"Labels: {labels}")
                    st.audio(audio, sample_rate=SAMPLE_RATE)

def get_subsets(sound_classes, df, results, N=15):

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
    parser.add_argument("--path0", type=str, default=None, 
                        help='Similarity Result Path 0.')
    parser.add_argument("--path1", type=str, default=None, 
                        help='Similarity Result Path 1.')
    parser.add_argument("--path2", type=str, default=None, 
                        help='Similarity Result Path 2.')
    parser.add_argument("--path3", type=str, default=None, 
                        help='Similarity Result Path 3.')
    parser.add_argument('-N', type=int, default=15, 
                        help="Number of top entries to display.")
    args=parser.parse_args()

    paths = [path for path in [args.path0, args.path1, args.path2, args.path3] if path is not None]

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