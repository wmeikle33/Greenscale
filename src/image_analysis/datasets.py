from pathlib import Path
import numpy as np
import tensorflow as tf


def load_25d_feature(path, image_size=(256, 256)):
    x = np.load(path).astype("float32")

    x = tf.image.resize(x, image_size)

    # normalize each channel safely
    max_vals = tf.reduce_max(x, axis=[0, 1], keepdims=True)
    x = tf.where(max_vals > 0, x / max_vals, x)

    return x


def load_mask(path, image_size=(256, 256)):
    y = np.load(path).astype("float32")

    if y.ndim == 2:
        y = y[..., None]

    y = tf.image.resize(y, image_size)
    y = tf.cast(y > 0.5, tf.float32)

    return y


def make_segmentation_dataset(
    features_dir,
    masks_dir,
    batch_size=8,
    image_size=(256, 256),
    shuffle=True,
):
    features_dir = Path(features_dir)
    masks_dir = Path(masks_dir)

    feature_paths = sorted(features_dir.glob("*_25d.npy"))

    pairs = []
    for feature_path in feature_paths:
        stem = feature_path.name.replace("_25d.npy", "")
        mask_path = masks_dir / f"{stem}_mask.npy"

        if mask_path.exists():
            pairs.append((str(feature_path), str(mask_path)))

    if not pairs:
        raise ValueError("No matching feature/mask pairs found.")

    feature_paths, mask_paths = zip(*pairs)

    ds = tf.data.Dataset.from_tensor_slices((list(feature_paths), list(mask_paths)))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(feature_paths))

    def _load(feature_path, mask_path):
        x = tf.numpy_function(
            lambda p: load_25d_feature(p.decode(), image_size),
            [feature_path],
            tf.float32,
        )
        y = tf.numpy_function(
            lambda p: load_mask(p.decode(), image_size),
            [mask_path],
            tf.float32,
        )

        x.set_shape((*image_size, None))
        y.set_shape((*image_size, 1))

        return x, y

    return ds.map(_load).batch(batch_size).prefetch(tf.data.AUTOTUNE)
