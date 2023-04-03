from argparse import ArgumentParser
from omegaconf import OmegaConf

import torch
from src.model import Wav2Vec2ForCTC
from pytorch_lightning import Trainer
from src.datamodule import VLSPDataModule

from src.utils import instantiate


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/wav2vec2-base-ctc.yaml")

    return parser.parse_args()

def main(args):
    config = OmegaConf.load(args.config)

    # Instantiate model
    model = Wav2Vec2ForCTC(**config.model)

    # Instantiate data
    datamodule = VLSPDataModule(**config.data)

    # Instantiate loggers
    if "loggers" in config.trainer:
        loggers = [instantiate(logger) for logger in config.trainer.pop("loggers")]
    else:
        loggers = None

    # Instantiate callbacks
    if "callbacks" in config.trainer:
        callbacks = [instantiate(callback) for callback in config.trainer.pop("callbacks")]
    else:
        callbacks = None

    # Instantiate trainer
    trainer = Trainer(logger=loggers, callbacks=callbacks, **config.pop("trainer"))
    
    # Start training!
    try:
        model = torch.compile(model, mode="default")
    except Exception as e:
        print("Cannot using torch.compile to accelerate training!")
        print(e)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    args = parse_args()
    main(args)
