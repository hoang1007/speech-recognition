import sys
sys.path.append("..")

from argparse import ArgumentParser
import os, string
from transformers import (
    Wav2Vec2ForPreTraining,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.datamodule import VLSP2020TarDataset
from src.datamodule.vlsp2020 import get_dataloader
from finetuning.wav2vec2 import SpeechRecognizer


def remove_punctuation(text: str):
    return text.translate(str.maketrans("", "", string.punctuation)).lower()


def prepare_dataloader(data_dir, batch_size, num_workers):
    train_dataset = VLSP2020TarDataset(
        os.path.join(data_dir, "vlsp2020_train_set.tar")
    ).load()
    val_dataset = VLSP2020TarDataset(
        os.path.join(data_dir, "vlsp2020_val_set.tar")
    ).load()

    train_dataloader = get_dataloader(
        train_dataset,
        return_transcript=True,
        target_transform=remove_punctuation,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_dataloader = get_dataloader(
        val_dataset,
        return_transcript=True,
        target_transform=remove_punctuation,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader


def prepare_model(adam_config: dict, tristate_scheduler_config: dict):
    model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"

    wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    model = SpeechRecognizer(
        wav2vec2, tokenizer, feature_extractor, adam_config, tristate_scheduler_config
    )

    return model


def main():
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--wav2vec2_lr", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=float, default=0.1)
    parser.add_argument("--constant_steps", type=float, default=0.4)
    parser.add_argument("--scheduler_factor", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--ckpt_path", type=str, default=None)

    args = parser.parse_args()
    print(args)

    train_loader, val_loader = prepare_dataloader(
        args.data_dir, args.batch_size, args.num_workers
    )

    total_steps = args.max_epochs * 42_000 // args.batch_size
    warmup_steps = int(total_steps * args.warmup_steps)
    constant_steps = int(total_steps * args.constant_steps)

    model = prepare_model(
        {
            "wav2vec2_lr": args.wav2vec2_lr,
            "classifier_lr": args.classifier_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "warmup_steps": warmup_steps,
            "constant_steps": constant_steps,
            "total_steps": total_steps,
            "factor": args.scheduler_factor,
        },
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        callbacks=[
            ModelCheckpoint(args.ckpt_dir, monitor="val/wer", mode="min", save_top_k=1),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=WandbLogger(project="Wav2Vec2"),
        max_epochs=args.max_epochs,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()