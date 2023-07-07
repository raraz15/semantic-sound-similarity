import os
import re

def get_fname(audio_path):
    """Returns the file name without the extension."""
    return os.path.splitext(os.path.basename(audio_path))[0]

def get_labels_of_fname(fname, df):
    """Returns the set of labels of the fname from the dataframe."""

    return set(df[df["fname"]==int(fname)]["labels"].values[0].split(","))

def get_all_labels(df):
    """Returns the set of all the labels in the dataframe."""

    return set([l for ls in df["labels"].apply(lambda x: x.split(",")).to_list() for l in ls])

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