import os
from difflib import SequenceMatcher
from glob import glob
from typing import Tuple

from torch import Tensor
from tqdm import tqdm

from clmr.datasets import Dataset


def similar(str1, str2, max_diff=1):
    """
    Checks if two strings are similar within a specified number of differences.

    Args:
        str1: The first string.
        str2: The second string.
        max_diff: The maximum number of allowed differences (default: 2).

    Returns:
        True if the strings are similar, False otherwise.
    """
    if str1 == str2:
        return True
    matcher = SequenceMatcher(None, str1, str2)
    return matcher.quick_ratio() >= 1 - max_diff / len(str1)


class AUDIO(Dataset):
    """Create a Dataset for any folder of audio files.
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        src_ext_audio (str): The extension of the audio files to analyze.
    """

    def __init__(
        self,
        root: str,
        src_ext_audio: str = ".wav",
        n_classes: int = 1,
        filenames_file: str = "",
    ) -> None:
        super(AUDIO, self).__init__(root)

        self._path = root
        self._src_ext_audio = src_ext_audio
        self.n_classes = n_classes

        if not filenames_file:
            self.fl = glob(
                os.path.join(self._path, "**", "*{}".format(self._src_ext_audio)),
                recursive=True,
            )
            if len(self.fl) == 0:
                raise RuntimeError(
                    "Dataset not found. Please place the audio files in the {} folder.".format(
                        self._path
                    )
                )
        else:
            with open(filenames_file, "r") as f:
                filenames = f.read().splitlines()
                self.fl = [os.path.join(self._path, f) for f in filenames]
                # check all files exist
                for file in self.fl:
                    if not os.path.isfile(file):
                        raise RuntimeError(
                            "File {} not found. Please place the audio files in the {} folder.".format(
                                file, self._path
                            )
                        )

        print(">>> Number of files in dataset:", len(self.fl))

    def file_path(self, n: int) -> str:
        fp = self.fl[n]
        return fp

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple [Tensor, Tensor]: ``(waveform, label)``
        """
        audio, _ = self.load(n)
        label = []
        return audio, label

    def __len__(self) -> int:
        return len(self.fl)
