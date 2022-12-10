import sys
sys.path.append(".")

from src.config import model as conf
from src.model import Wav2Vec2PretrainingModule
from src.datamodule import WebDatasetConverter, VLSP2020ForPretrainingDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == "__main__":

    model = Wav2Vec2PretrainingModule(conf.wav2vec2_pretraining)
    dts = WebDatasetConverter(conf.dataset.path).get_dataset()
    dtm = VLSP2020ForPretrainingDataModule(dts, **conf.dataset)
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                dirpath=conf["checkpoint_dir"],
            )
        ],
        gradient_clip_val=1.0,
    )

    trainer.fit(model, dtm)
