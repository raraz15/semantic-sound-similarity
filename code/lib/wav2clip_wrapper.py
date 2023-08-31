import torch
import numpy as np

from wav2clip.model.encoder import ResNetExtractor

def get_model(model_path, device="cpu", pretrained=True, frame_length=None, hop_length=None):
    if pretrained:
        checkpoint = torch.load(model_path, map_location=device)
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     MODEL_URL, map_location=device, progress=True
        # )
        model = ResNetExtractor(
            checkpoint=checkpoint,
            scenario="frozen",
            transform=True,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    else:
        model = ResNetExtractor(
            scenario="supervise", frame_length=frame_length, hop_length=hop_length
        )
    model.to(device)
    return model


def embed_audio(audio, model):
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=0)
    return (
        model(torch.from_numpy(audio).to(next(model.parameters()).device))
        .detach()
        .cpu()
        .numpy()
    )