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

def sample_sound(df, similarity_dict, sound_names, N=15):

    idx = random.randint(0,len(similarity_dict))
    fname = sound_names[idx]
    labels =  df[df.fname==int(fname)].labels.values[0]
    similarities = similarity_dict[fname]
    audio_path = os.path.join(AUDIO_DIR, f"{fname}.wav")
    with st.container():
        st.subheader("Query Sound")
        st.caption(f"Sound ID: {fname}")
        st.caption(f"Labels: {labels}")
        audio = MonoLoader(filename=audio_path,
                            sampleRate=SAMPLE_RATE)()
        st.audio(audio, sample_rate=SAMPLE_RATE)
    st.divider()
    st.subheader(f"Top {N} Similar Sounds")
    with st.container():
        columns = st.columns(3)
        for i, result in enumerate(similarities[:N]):
            fname = list(result.keys())[0]
            labels = df[df.fname==int(fname)].labels.values[0]
            #if len(labels) < max_length:
            #    labels += " "*(max_length-len(labels))
            audio_path = os.path.join(AUDIO_DIR, f"{fname}.wav")
            audio = MonoLoader(filename=audio_path,
                                sampleRate=SAMPLE_RATE)()
            with columns[i//5]:
                st.write(f"Ranking: {i+1}")
                #st.caption(f"Sound ID: {fname}")
                st.caption(f"Labels: {labels}")
                st.audio(audio, sample_rate=SAMPLE_RATE)
                #st.divider()
    return similarities

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
    sound_names = list(similarity_dict.keys())

    # Read the meta data
    df = pd.read_csv(GT_PATH)
    df.labels = df.labels.apply(lambda x: re.sub("_", " ", re.sub(",", ", ", x)))
    #max_length = max([len(x) for x in df.labels.to_list()])

    st.title("Freesound Sound Similarity Results")
    st.text(f"Embeddings: {embeddings}")
    st.text(f"Search Method: {search}")
    st.text(f"Dataset: FSD50K.eval_audio")
    st.header("Click To Sample a Random Sound and Display Similar Sounds.")

    st.button(label=":loud_sound:", 
            key="sample", 
            on_click=sample_sound, 
            args=(df, similarity_dict, sound_names), 
            kwargs={"N":args.N}
            )