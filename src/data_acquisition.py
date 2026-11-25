"""Data acquisition: download the NIH malaria cell images and write them to
data/train/<Class>/ and data/test/<Class>/ (PNG files).

Split matches the notebook: first 85% of the dataset -> data/train (the notebook
further cuts 15% of the total off it as validation), last 15% -> data/test.

Usage (from the project root):  python -m src.data_acquisition
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["Parasitized", "Uninfected"]


def main():
    import tensorflow_datasets as tfds

    cache = ROOT / "data" / "_tfds_cache"
    splits = {"train": "train[:85%]", "test": "train[85%:]"}
    ds_train, ds_test = tfds.load("malaria", split=list(splits.values()), as_supervised=True,
                                  data_dir=str(cache))
    for name, ds in (("train", ds_train), ("test", ds_test)):
        for cls in CLASS_NAMES:
            (ROOT / "data" / name / cls).mkdir(parents=True, exist_ok=True)
        counts = [0, 0]
        for i, (img, lab) in enumerate(tfds.as_numpy(ds)):
            lab = int(lab)
            Image.fromarray(img).save(ROOT / "data" / name / CLASS_NAMES[lab] / f"{i:06d}.png")
            counts[lab] += 1
        print(name, dict(zip(CLASS_NAMES, counts)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
