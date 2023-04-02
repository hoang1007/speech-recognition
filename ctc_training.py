from argparse import ArgumentParser
from omegaconf import OmegaConf

import torch
from src.model import Wav2Vec2ForCTC
from pytorch_lightning import Trainer
from src.datamodule import VLSPDataModule

from src.utils import instantiate


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, default="configs/wav2vec2-base-ctc.yaml")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])

    return parser.parse_args()

def main(args):
    config = OmegaConf.load(args.config)

    # Instantiate model
    model = Wav2Vec2ForCTC(**config.model)

    # Instantiate data
    datamodule = VLSPDataModule(**config.data)

    # Instantiate loggers
    logger = [instantiate(logger) for logger in config.pop("logger")]

    # Instantiate callbacks
    callbacks = [instantiate(callback) for callback in config.pop("callbacks")]

    # Instantiate trainer
    trainer = Trainer(logger=logger, callbacks=callbacks, **config.pop("trainer"))
    
    # Start training!
    model = torch.compile(model, mode="default")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    args = parse_args()
    main(args)
