# Data layout

```text
data/archive/                  immutable source archives
data/classification/           generated train, validation, and test splits
data/demo/                     tray images and uploads used by the demo
data/extras/                   additional reviewed source images
data/inbox/review/             Data IDE review queue
data/reviewed/                 accepted source images by dish class
```

The runtime and training workflow use image-level dish labels only. Generated datasets can be rebuilt from reviewed sources with `scripts/data/01_build_classification_dataset.py`.

Cloud packages are written to `outputs/cloud/classification.zip` with a matching manifest. Classifier runs are stored under `runs/classifier/<timestamp>/` locally or in the configured Drive root.
