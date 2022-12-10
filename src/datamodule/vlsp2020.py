import os
import json
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import webdataset


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

        return transcript, audio


class WebDatasetConverter:
    def __init__(self, outpath: str):
        self.outpath = outpath

    def convert(self, *args, **kwargs):
        import sys

        self.dts = VLSP2020Dataset(*args, **kwargs)

        writer = webdataset.TarWriter(self.outpath)

        for idx, (transcript, audio) in enumerate(self.dts):
            writer.write(
                {
                    "__key__": f"{idx:08d}",
                    "txt": transcript,
                    "pth": audio,
                }
            )

            if idx % 1000 == 0:
                print(f"{idx:6d}", end="\r", flush=True, file=sys.stderr)

        writer.close()

    def get_dataset(self):
        return (
            webdataset.WebDataset(self.outpath)
            .decode(
                webdataset.handle_extension("txt", lambda x: x.decode("utf-8")),
                webdataset.torch_audio,
            )
            .to_tuple("txt", "pth")
        )


class VLSP2020ForPretrainingDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        train_ratio: str = 0.75,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data = dataset
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers

        self.save_hyperparameters()

    def setup(self, stage: str = None) -> None:
        # self.train_data, self.val_data = random_split(self.data, [600, 200])
        self.train_data = self.data

    @staticmethod
    def collate_fn(batch):
        # item[1] is audio
        audio = tuple(item[1] for item in batch)

        return audio

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
