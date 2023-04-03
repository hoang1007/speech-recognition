# from src.model import Wav2Vec2PretrainingModule
from torch.hub import load_state_dict_from_url
from torch.nn import Module

def load_pretrained_model(model: Module):
    """
    Mapping weights from LeBinh's Wav2Vec2 pretrained model to model in this repo.
    The pretrained model can be found here: https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h
    """
    print("Loading pretrained weights...")
    pretrained_url = "https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h/resolve/main/pytorch_model.bin"
    ckpt = load_state_dict_from_url(pretrained_url, map_location="cpu")
    
    # Load the model
    state_dict = mapping_weights(model, ckpt)
    model.load_state_dict(state_dict, strict=False)

    return model

def mapping_weights(model: Module, ckpt):
    model_state_dict = dict()

    for src_param_name, weight in ckpt.items():
        if "feature_extractor" in src_param_name:
            dst_param_name = src_param_name.replace("wav2vec2.", "")
        else:
            dst_param_name = src_param_name.replace("wav2vec2", "context_encoder")
        model_state_dict[dst_param_name] = weight
    
    # Find missing parameters
    dst_params = list(model.state_dict().keys())
    missing_params = set(dst_params) - set(model_state_dict.keys())
    print(f"Missing parameters: {missing_params}")

    return model_state_dict
