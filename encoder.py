import clip
import torch.cuda
import numpy as np
from torch import nn

# def decode_inst(inst):
#   """Utility to decode encoded language instruction"""
#   return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8") 

def decode_inst(inst):
    nonzero_indices = torch.nonzero(inst)
    nonzero_values = inst[nonzero_indices].tolist()
    tensor_values = torch.tensor(nonzero_values)
    byte_values = bytes(tensor_values)
    decoded_text = byte_values.decode("utf-8")
    return decoded_text


def text_encoder(instruction):           #clip textencoder
    # ls = []
    # ls.append(decode_inst(instruction))
    ls = decode_inst(instruction)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(ls).to(device)
    text_features = model.encode_text(text)    #[1,512]
    # print("text shape:",text_features.shape)
    
    return text_features


def rgb_encoder(rgb):                    #pretrained resnet18 ->50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    rgb = rgb.unsqueeze(0).permute(0,3,1,2).float().to(device)
    if torch.cuda.is_available():
        model.to('cuda')
    with torch.no_grad():
        rgb = model(rgb)                    # [1,1000]
        # print(rgb.shape)
    return rgb

