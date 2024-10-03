config = {
    # training
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "lr_patience": 10,
    "epochs": 1000,
    # dataset
    "dataset_dir": "./data/magnatagatune/",
    "filetype": "wav",
    "items_per_track": 8,
    "only_from_tag": None,
    "tracks_per_genre": 0,
    "random_percentage": None,
}
