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

def display_query_and_similar_sound(fname, df, similarity_dict, N=15, header=None):

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
        columns = st.columns(3)
        for i, result in enumerate(similarity_dict[fname][:N]):
            fname = list(result.keys())[0]
            labels = df[df.fname==int(fname)].labels.values[0]
            audio_path = os.path.join(AUDIO_DIR, f"{fname}.wav")
            audio = MonoLoader(filename=audio_path,
                                sampleRate=SAMPLE_RATE)()
            with columns[i//5]:
                st.write(f"Ranking: {i+1} - Score: {list(result.values())[0]:.3f}")
                st.caption(f"Labels: {labels}")
                st.audio(audio, sample_rate=SAMPLE_RATE)

def get_subsets(sound_classes, df, similarity_dict, N=15):

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
    display_query_and_similar_sound(fname, df, similarity_dict, N=N, header=header)

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='Similarity Result Path.')
    parser.add_argument('-N', type=int, default=15, 
                        help="Number of top entries to display.")
    args=parser.parse_args()

    search = os.path.basename(os.path.dirname(args.path))
    if search=="nn":
        search = "Nearest Neighbor"
    elif search=="dot":
        search = "Dot Product"
    embeddings = os.path.basename(os.path.dirname(os.path.dirname(args.path)))

    # Load the analysis file
    with open(args.path ,"r") as infile:
        similarity_dict = json.load(infile)
    fnames = list(similarity_dict.keys())

    # Read the meta data
    df = pd.read_csv(GT_PATH)
    df.labels = df.labels.apply(lambda x: re.sub("_", " ", re.sub(",", ", ", x)))
    all_labels = sorted(list(set([y for x in df.labels.to_list() for y in x.split(", ")])))

    st.title("Freesound Sound Similarity Results")
    st.text(f"Embeddings: {embeddings} - Search Method: {search} - Dataset: FSD50K.eval_audio")

    st.header("Choose Sound Categories and Click The Speaker Icon.")
    sound_classes = st.multiselect("Sound Classes", options=all_labels)
    st.button(label=":speaker:", 
            on_click=get_subsets, 
            args=(sound_classes, df, similarity_dict), 
            kwargs={"N":args.N}
            )