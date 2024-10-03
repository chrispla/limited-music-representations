config = {
    # training
    "batch_size": 24,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "lr_patience": 10,
    "epochs": 150,
    # dataset
    "dataset_dir": "/home/chris/dev/metamae/music-MetaMAE/datasets/music/MTAT/",
    "filetype": "wav",
    "items_per_track": 6,
    "only_from_tag": 0,  # 0 for all tracks
    "tracks_per_genre": 0,  # 0 for all tracks
    "random_percentage": 0,  # 0 for all tracks
}
