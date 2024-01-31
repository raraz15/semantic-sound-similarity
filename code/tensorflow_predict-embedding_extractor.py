"""Takes a FSD50K csv file specifying audio file names and computes embeddings 
using a TensorFlowPrecict model. All frame embeddings are exported without 
aggregation. Currently works with FSD-Sinet, VGGish, YamNet and OpenL3 models."""

import os
import time
import json
import glob
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

from essentia.standard import EasyLoader, TensorflowPredictFSDSINet, TensorflowPredictVGGish

from lib.openl3 import EmbeddingsOpenL3
from lib.directories import EMBEDDINGS_DIR

TRIM_DUR = 30 # seconds

def create_frame_level_embeddings(model, audio, model_name):
    """ Takes an embedding model and an audio array and returns the frame level embeddings.
    If the model produces a non-floatable embedding, returns None. This does not happen
    with models such as FSD-Sinet or VGGish, YamNet, OpenL3 on FSD50K eval."""

    try:
        # Embeddings of each time frame
        if "openl3" in model_name:
            embeddings = model.compute(audio) 
        else:
            embeddings = model(audio)
        # Convert to list of lists, float for json serialization
        embeddings = [[float(value) for value in embedding] for embedding in embeddings]
        return embeddings
    except AttributeError:
        print("Model produced a non-floatable embedding.")
        return None

def process_audio(model, audio_path, output_dir, sample_rate, model_name):
    """ Reads the audio of given path, creates the embeddings and exports."""
    # Load the audio file
    loader = EasyLoader()
    loader.configure(filename=audio_path, 
                     sampleRate=sample_rate, 
                     endTime=TRIM_DUR, # FSD50K are already below 30 seconds
                     replayGain=0 # Do not normalize the audio
                     )
    audio = loader()
    # Zero pad short clips (IN FSD50K 7% of the clips are shorter than 1 second)
    if audio.shape[0] < sample_rate:
        audio = np.concatenate((audio, np.zeros((sample_rate-audio.shape[0]))))
    # Process
    embeddings = create_frame_level_embeddings(model, audio, model_name)
    # Save results
    fname = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{fname}.json")
    with open(output_path, 'w') as outfile:
        json.dump({'audio_path': audio_path, 
                   'embeddings': embeddings}, outfile, indent=4)

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_path',
                        type=str, 
                        help="Path to config.json file of the model. "
                        "Assumes the model.pb is next to it.")
    parser.add_argument('audio_dir',
                        type=str,
                        help="Path to directory with audio files.")
    parser.add_argument('-o', 
                        '--output_dir', 
                        type=str, 
                        default="",
                        help="Path to output directory.")
    args=parser.parse_args()

    # Read the config file
    with open(args.config_path, "r") as json_file:
        config = json.load(json_file)
    print("Config:")
    print(json.dumps(config, indent=4))

    # Configure the embedding model
    model_name = os.path.splitext(os.path.basename(args.config_path))[0]
    model_path = os.path.join(os.path.dirname(args.config_path), f"{model_name}.pb")
    if "audioset-yamnet" in model_name:
        model = TensorflowPredictVGGish(graphFilename=model_path, 
                                        input="melspectrogram", 
                                        output="embeddings")
    elif "audioset-vggish" in model_name:
        model = TensorflowPredictVGGish(graphFilename=model_path, 
                                        output="model/vggish/embeddings")
    elif "fsd-sinet" in model_name:
        model = TensorflowPredictFSDSINet(graphFilename=model_path,
                                        output="model/global_max_pooling1d/Max")
    elif "openl3" in model_name:
        model = EmbeddingsOpenL3(graph_path=model_path,
                                input_shape=config['schema']['inputs'][0]['shape'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Get the list of audio files
    args.audio_dir = os.path.normpath(args.audio_dir)
    audio_paths = glob.glob(os.path.join(args.audio_dir, "*.wav"))
    assert len(audio_paths)>0, f"No audio files found in {args.audio_dir}."
    print(f"Found {len(audio_paths)} audio files in {args.audio_dir}.")

    # Determine the output directory
    if args.output_dir=="":
        # If the default output directory is used add the args.audio_dir to the path
        output_dir = os.path.join(EMBEDDINGS_DIR,
                                os.path.basename(args.audio_dir),
                                model_name)
    else:
        output_dir = os.path.join(args.output_dir, model_name)
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting the embeddings to: {output_dir}")

    # Process each audio
    start_time = time.time()
    for i,audio_path in enumerate(audio_paths):
        process_audio(model, 
                    audio_path, 
                    output_dir, 
                    config['inference']['sample_rate'], 
                    model_name)
        if i%1000==0 or i+1==len(audio_paths) or i==0:
            print(f"[{i+1:>{len(str(len(audio_paths)))}}/{len(audio_paths)}]")
    total_time = time.time()-start_time
    print(f"\nTotal time: {time.strftime('%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

    #############
    print("Done!\n")