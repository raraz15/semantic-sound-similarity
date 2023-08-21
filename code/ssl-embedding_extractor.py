"""This script load a self-supervised model with a pre-trained checkpoint and
extracts clip level embeddings for all the audio files in the FSD50K evaluation 
dataset."""

import os
import time
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd

import torch

from lib.directories import AUDIO_DIR, GT_PATH, EMBEDDINGS_DIR

TRIM_DUR = 30 # seconds

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path',
                        type=str, 
                        help="Path to model.pt chekpoint.")
    parser.add_argument('-o', 
                        '--output_dir', 
                        type=str, 
                        default="",
                        help="Path to output directory.")
    args=parser.parse_args()

    # Get the model anem from models/model_name.pt
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    # Load the corresponding model
    if "clap" in model_name.lower():
        print("Setting up CLAP model...")
        from lib.laion_clap import CLAP_Module
        # Decide type of CLAP model
        if model_name in ["clap-630k-audioset-fusion-best", "clap-630k-fusion-best"]:
            model = CLAP_Module(enable_fusion=True)
        elif "clap-music_speech_audioset_epoch_15_esc_89.98" == model_name:
            model= CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
        else:
            raise ValueError(f"Unknown CLAP model name: {model_name}")
        # Load the model
        model.load_ckpt(args.model_path)
        # Define embedding extractor function
        def extract_embeddings(model, audio_path):
            # Process
            embeddings = model.get_audio_embedding_from_filelist(x=[audio_path], 
                                                                    use_tensor=False).tolist()
            return embeddings
    elif "imagebind" in model_name.lower():
        print("Setting up ImageBind model...")
        from lib.imagebind import data
        from lib.imagebind.models import imagebind_model
        from lib.imagebind.models.imagebind_model import ModalityType        
        # Load the model
        model = imagebind_model.imagebind_huge(args.model_path, pretrained=True)
        model.eval()
        model.to("cpu")
        # Define embedding extractor function
        def extract_embeddings(model, audio_path):
            # Load the audio file
            inputs = {
                ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], 'cpu'),
            }
            # Process
            with torch.no_grad():
                embeddings = model(inputs)
            embeddings = embeddings['audio'].cpu().numpy().tolist()
            return embeddings
    elif "audioclip" in model_name.lower():
        print("Setting up AudioCLIP model...")        # TODO: create a function to load audio files, use it in tensofrlowpredict script as well
        from lib.audio_clip.model import AudioCLIP
        from lib.audio_clip.utils.transforms import ToTensor1D
        import librosa
        # Load the model
        model = AudioCLIP(pretrained=args.model_path).eval()
        # Define embedding extractor function
        def extract_embeddings(model, audio_path):
            # Load the audio file
            audio = librosa.load(audio_path, sr=44100)[0]
            # Trim the audio
            audio = audio[:TRIM_DUR*44100]
            # Bring to the right format
            audio = audio.astype(np.float32)
            audio_transforms = ToTensor1D()
            audio = torch.stack([audio_transforms(audio.reshape(1,-1))])
            # Process
            ((embeddings, _, _), _), _ = model(audio=audio)
            embeddings = embeddings.tolist()
            return embeddings
    elif "wav2clip" in model_name.lower():
        print("Setting up wav2clip model...")
        import lib.wav2clip_wrapper as wav2clip
        import librosa
        # Load the model
        model = wav2clip.get_model(args.model_path)
        # Define embedding extractor function
        def extract_embeddings(model, audio_path):
            # Load the audio file
            audio = librosa.load(audio_path, sr=44100)[0]
            # Trim the audio
            audio = audio[:TRIM_DUR*44100]
            # Create the embeddings
            embeddings = wav2clip.embed_audio(audio, model).tolist()
            return embeddings
    else:
        raise ValueError(f"Unknown model name: {model_name}.")

    # Read the file names
    fnames = pd.read_csv(GT_PATH)["fname"].to_list()
    audio_paths = []
    for fname in fnames:
        audio_path = os.path.join(AUDIO_DIR, f"{fname}.wav")
        if os.path.exists(audio_path):
            audio_paths.append(audio_path)
    assert len(audio_paths)==len(fnames), "Some audio files are missing."
    print(f"There are {len(audio_paths)} audio files to process.")

    # Determine the output directory
    if args.output_dir=="":
        # If the default output directory is used add the AUDIO_DIR to the path
        output_dir = os.path.join(EMBEDDINGS_DIR, 
                                os.path.basename(AUDIO_DIR),
                                model_name)
    else:
        output_dir = os.path.join(args.output_dir, model_name)
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting the embeddings to: {output_dir}")

    # Process each audio clip
    start_time = time.time()
    for i,audio_path in enumerate(audio_paths):
        # Extract the embeddings
        embeddings = extract_embeddings(model, audio_path)
        # Save results
        fname = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{fname}.json")
        with open(output_path, 'w') as outfile:
            json.dump({'audio_path': audio_path, 'embeddings': embeddings}, outfile, indent=4)
        # Print progress
        if (i+1)%1000==0 or i==0 or i+1==len(audio_paths):
            print(f"[{i+1:>{len(str(len(audio_paths)))}}/{len(audio_paths)}]")        
    total_time = time.time()-start_time
    print(f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

    #############
    print("Done!\n")    