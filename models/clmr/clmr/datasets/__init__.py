import os

# fmt: off
from .dataset import Dataset

h = 2
from .audio import AUDIO
from .librispeech import LIBRISPEECH

h = 1  # im doing this temporarily to disable black putting GTZAN over LIBRISPEECH,
# which causes a circular import error

from .gtzan import GTZAN
from .magnatagatune import MAGNATAGATUNE
from .million_song_dataset import MillionSongDataset

# fmt: on


def get_dataset(dataset, dataset_dir, subset, download=True, args=None):

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "audio":
        d = AUDIO(root=dataset_dir, filenames_file=args.filenames_file)
    elif dataset == "librispeech":
        d = LIBRISPEECH(root=dataset_dir, download=download, subset=subset)
    elif dataset == "gtzan":
        d = GTZAN(root=dataset_dir, download=download, subset=subset)
    elif dataset == "magnatagatune":
        d = MAGNATAGATUNE(root=dataset_dir, download=download, subset=subset)
    elif dataset == "msd":
        d = MillionSongDataset(root=dataset_dir, subset=subset)
    else:
        raise NotImplementedError("Dataset not implemented")
    return d
