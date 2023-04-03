from warnings import warn
from typing import Callable, Optional, Union
from tqdm import tqdm
import os
import string

import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
import torch_audiomentations as T

from pytorch_lightning import LightningDataModule

try:
    import webdataset
except ImportError as e:
    webdataset = None
    warn("WebDataset is not installed. Please install it to use compressed dataset.")

metadata = dict(
    vlsp2020=dict(
        num_samples=56427
    )
)

class VLSP2020Dataset(Dataset):
    def __init__(self, root: str, sample_rate: int = 16000):
        super().__init__()

        self.sample_rate = sample_rate
        self.memory = self._prepare_data(root)
        self._memory = tuple(
            (v["transcript"], v["audio"]) for v in self.memory.values()
        )

    @staticmethod
    def _prepare_data(root: str):
        memory = {}

        for f in os.scandir(root):
            file_name, file_ext = os.path.splitext(f.name)

            if file_ext == ".txt":
                if file_name not in memory:
                    memory[file_name] = {"transcript": f.path}
                elif "transcript" not in memory[file_name]:
                    memory[file_name]["transcript"] = f.path
                else:
                    raise ValueError(f"Duplicate transcript for {f.path}")
            else:
                if file_name not in memory:
                    memory[file_name] = {"audio": f.path}
                elif "audio" not in memory[file_name]:
                    memory[file_name]["audio"] = f.path
                else:
                    raise ValueError(f"Duplicate audio for {f.path}")

        for key, value in memory.items():
            if "audio" not in value:
                raise ValueError(f"Missing audio for {key}")
            elif "transcript" not in value:
                raise ValueError(f"Missing transcript for {key}")

        return memory

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index: int):
        transcript, audio = self._memory[index]

        with open(transcript, "r") as f:
            transcript = f.read()

        audio, sample_rate = torchaudio.load(audio)
        audio = F.resample(audio, sample_rate, self.sample_rate)

        return audio, transcript


class VLSP2020TarDataset:
    def __init__(self, outpath: str):
        self.outpath = outpath

    def convert(self, dataset: VLSP2020Dataset):
        assert webdataset is not None, "WebDataset is not installed."
        writer = webdataset.TarWriter(self.outpath)

        for idx, (audio, transcript) in enumerate(tqdm(dataset, colour="green")):
            writer.write(
                {
                    "__key__": f"{idx:08d}",
                    "txt": transcript,
                    "pth": audio,
                }
            )

        writer.close()

    def load(self):
        assert webdataset is not None, "WebDataset is not installed."
        self.data = (
            webdataset.WebDataset(self.outpath)
            .decode(
                webdataset.handle_extension("txt", lambda x: x.decode("utf-8")),
                webdataset.torch_audio,
            )
            .to_tuple("txt", "pth")
        )

        return self.data


class VLSPDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        train_val_ratio: float = 0.75,
        num_workers: int = 2,
        pin_memory: bool = True,
        sample_rate: int = 16000,
        return_transcript: bool = False,
    ):
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sample_rate = sample_rate
        self.return_transcript = return_transcript

    def prepare_data(self):
        dataset = VLSP2020Dataset(self.root, self.sample_rate)
        if webdataset is None:
            warn("WebDataset is not installed. Please install it to use compressed dataset.")
        else:
            dataset = VLSP2020TarDataset(os.path.join(self.root, "vlsp2020.tar")).convert(dataset)

    def setup(self, stage: Optional[str] = None):
        if webdataset is None:
            dataset = VLSP2020Dataset(self.root, self.sample_rate)
        else:
            dataset = VLSP2020TarDataset(os.path.join(self.root, "vlsp2020.tar")).load()

        num_training_samples = int(metadata["vlsp2020"]["num_samples"] * self.train_val_ratio)
        num_validation_samples = metadata["vlsp2020"]["num_samples"] - num_training_samples
        self.train_data, self.val_data = random_split(dataset, [num_training_samples, num_validation_samples])

    def get_dataloader(
        self,
        dataset,
        return_transcript: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs
    ):
        def collate_fn(batch):
            def get_audio(item):
                audio = item[0]

                if transform is not None:
                    audio = transform(audio, self.sample_rate)

                assert (
                    isinstance(audio, torch.Tensor)
                    and audio.ndim == 2
                    and audio.size(0) == 1
                )

                return audio.squeeze(0)

            audio = tuple(get_audio(item) for item in batch)

            if return_transcript:
                if target_transform is not None:
                    transcript = tuple(target_transform(item[0]) for item in batch)
                else:
                    transcript = tuple(item[1] for item in batch)

                return audio, transcript
            else:
                return audio

        return DataLoader(
            dataset, collate_fn=collate_fn, **kwargs
        )

    def train_dataloader(self):
        transform = T.Compose([
            T.Gain(min_gain_in_db=-5.0, max_gain_in_db=5.0, p=0.5),
            T.PolarityInversion(p=0.5),
            T.AddColoredNoise(p=0.2),
        ])
        target_transform = lambda text: text.translate(str.maketrans("", "", string.punctuation)).lower()

        return self.get_dataloader(
            self.train_data,
            self.return_transcript,
            transform=transform,
            target_transform=target_transform,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        target_transform = lambda text: text.translate(str.maketrans("", "", string.punctuation)).lower()

        return self.get_dataloader(
            self.val_data,
            self.return_transcript,
            target_transform=target_transform,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
