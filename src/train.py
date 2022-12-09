from config import model as conf
from modules.wav2vec2_wrapper import Wav2Vec2PretrainingModule

import torch

model = Wav2Vec2PretrainingModule(conf.wav2vec2_pretraining)

waveforms = (
    torch.rand(20000),
    torch.rand(25000),
)

loss = model(waveforms)