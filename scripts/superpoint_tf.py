#!/usr/bin/env python3
import argparse
from itertools import chain
from pathlib import Path
import os

# disable tensorflow C++ log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm


class NMSPool(layers.Layer):
    def __init__(self, nms_radius):
        super().__init__()
        self._nms_radius = nms_radius
        self._pool = layers.MaxPool2D(
            pool_size=(nms_radius * 2 + 1, nms_radius * 2 + 1),
            strides=(1, 1),
            padding="valid",
        )

    def call(self, x):
        pad_x = tf.pad(
            x,
            paddings=tf.constant(
                [
                    [0, 0],
                    [self._nms_radius, self._nms_radius],
                    [self._nms_radius, self._nms_radius],
                ]
            ),
        )
        return tf.squeeze(self._pool(pad_x[:, :, :, tf.newaxis]), axis=-1)


class SuperPointTF(tf.keras.Model):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.pool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.relu = layers.ReLU()

        self.conv1a = layers.Conv2D(
            c1,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, 1),
        )
        self.conv1b = layers.Conv2D(
            c1,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, c1),
        )
        self.conv2a = layers.Conv2D(
            c2,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, c1),
        )
        self.conv2b = layers.Conv2D(
            c2,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, c2),
        )
        self.conv3a = layers.Conv2D(
            c3,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, c2),
        )
        self.conv3b = layers.Conv2D(
            c3,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, c3),
        )
        self.conv4a = layers.Conv2D(
            c4,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, c3),
        )
        self.conv4b = layers.Conv2D(
            c4,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, c4),
        )

        self.convPa = layers.Conv2D(
            c5,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, c4),
        )
        self.convPb = layers.Conv2D(
            65,
            kernel_size=1,
            strides=(1, 1),
            padding="valid",
            input_shape=(None, None, c5),
        )

        self.convDa = layers.Conv2D(
            c5,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            input_shape=(None, None, c4),
        )
        self.convDb = layers.Conv2D(
            self.config["descriptor_dim"],
            kernel_size=1,
            strides=(1, 1),
            padding="valid",
            input_shape=(None, None, c5),
        )

        mk = self.config["max_keypoints"]
        if mk == 0 or mk < -1:
            raise ValueError('"max_keypoints" must be positive or "-1"')
        nms_radius = self.config["nms_radius"]
        if nms_radius <= 0:
            raise ValueError("nms_radius must be positive integer")
        self._nms_pool = NMSPool(nms_radius)

    def call(self, image):
        x = self.call_encoder(image)
        dense_descriptors = self.call_dense_descriptor(x)
        scores = self.call_scores(x)
        return x, scores, dense_descriptors

    def call_(self, image):
        _, scores, dense_descriptors = self(image)
        # Discard keypoints near the image borders
        _, height, width = scores.shape
        keypoints = [tf.where(s > self.config["keypoint_threshold"]) for s in scores]
        scores = [tf.gather_nd(s, indices=k) for s, k in zip(scores, keypoints)]
        keypoints, scores = list(
            zip(
                *[
                    self.remove_borders(k, s, height, width)
                    for k, s in zip(keypoints, scores)
                ]
            )
        )
        # Keep the k keypoints with highest score
        keypoints, scores = list(
            zip(*[self.top_k_keypoints(k, s) for k, s in zip(keypoints, scores)])
        )
        # Convert (h, w) to (x, y)
        keypoints = [tf.cast(k[:, ::-1], dtype=tf.float32) for k in keypoints]
        # Extract descriptors
        descriptors = [
            self.sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, dense_descriptors)
        ]
        return keypoints, scores, descriptors

    def call_encoder(self, image):
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        return self.relu(self.conv4b(x))

    def call_dense_descriptor(self, x):
        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        return tf.linalg.normalize(descriptors, ord=2, axis=-1)[0]

    def call_scores(self, x):
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = tf.nn.softmax(scores, axis=-1)[:, :, :, :-1]
        b, h, w, _ = scores.shape
        scores = tf.reshape(scores, (b, h, w, 8, 8))
        scores = tf.reshape(
            tf.transpose(scores, perm=[0, 1, 3, 2, 4]), (b, h * 8, w * 8)
        )
        return self.simple_nms(scores)

    def simple_nms(self, scores):
        zeros = tf.zeros_like(scores)
        max_mask = scores == self._nms_pool(scores)
        for _ in range(2):
            supp_mask = self._nms_pool(tf.cast(max_mask, dtype=tf.float32)) > 0
            supp_scores = tf.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == self._nms_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return tf.where(max_mask, scores, zeros)

    def load_torch_state_dict(self, state_dict):
        import torch

        if isinstance(state_dict, str):
            state_dict = {
                k: v.detach().cpu().numpy() for k, v in torch.load(state_dict).items()
            }
        dummy_input = tf.random.normal((1, 200, 200, 1), dtype=tf.float32)
        _ = self(dummy_input)
        for name in [
            "conv1a",
            "conv1b",
            "conv2a",
            "conv2b",
            "conv3a",
            "conv3b",
            "conv4a",
            "conv4b",
            "convDa",
            "convDb",
            "convPa",
            "convPb",
        ]:
            conv_layer = getattr(self, name)
            weight = state_dict[f"{name}.weight"].transpose([2, 3, 1, 0])
            bias = state_dict[f"{name}.bias"]
            conv_layer.set_weights([weight, bias])

    def remove_borders(self, keypoints, scores, height, width):
        """Removes keypoints too close to the border"""
        border = self.config["remove_borders"]
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
        mask = mask_h & mask_w
        return keypoints[mask], scores[mask]

    def top_k_keypoints(self, keypoints, scores):
        k = self.config["max_keypoints"]
        if k >= len(keypoints) or k <= 0:
            return keypoints, scores
        scores, indices = tf.math.top_k(scores, k)
        return tf.gather(keypoints, indices), scores

    def convert_tflite(
        self, img_shape, repr_dataset_paths, optimizations=None
    ) -> bytes:
        if optimizations is None:
            optimizations = [tf.lite.Optimize.DEFAULT]
        if optimizations:

            def representive_dataset():
                for dataset_path in repr_dataset_paths:
                    dataset = Path(dataset_path)
                    print(f"processing {dataset}")
                    for img_path in tqdm(
                        [p for p in chain(dataset.glob("*.png"), dataset.glob("*.jpg"))]
                    ):
                        np_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if np_gray.shape != img_shape:
                            print(
                                f"skipping {img_path} for invalid image shape: {np_gray.shape}"
                            )
                            continue
                        tf_inp = tf.convert_to_tensor(
                            np_gray.astype(np.float32)[None, :, :, None] / 255.0,
                            dtype=tf.float32,
                        )
                        yield [tf_inp]

        else:
            representive_dataset = None
        input_shape = (1,) + img_shape + (1,)

        @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32)])
        def superpoint_tf_func(image):
            x = self.call_encoder(image)
            return self.call_dense_descriptor(x)

        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [
                superpoint_tf_func.get_concrete_function(
                    tf.random.normal(input_shape, dtype=tf.float32)
                )
            ]
        )
        converter.optimizations = optimizations
        converter.representative_dataset = representive_dataset
        tflite_model = converter.convert()
        return tflite_model

    @staticmethod
    def sample_descriptors(keypoints: tf.Tensor, descriptors: tf.Tensor, s: int = 8):
        import torch

        # TODO: pure tensorflow grid_sampl impl
        # https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159
        # https://stackoverflow.com/questions/52888146/what-is-the-equivalent-of-torch-nn-functional-grid-sample-in-tensorflow-numpy
        """Interpolate descriptors at keypoint locations"""
        descriptors = tf.transpose(descriptors, [0, 3, 1, 2])
        b, c, h, w = descriptors.shape
        keypoints = keypoints - s / 2 + 0.5  # BxNx2
        keypoints /= tf.convert_to_tensor(
            [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)], dtype=keypoints.dtype
        )[
            tf.newaxis
        ]  # BxNx2
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        th_descriptors = torch.from_numpy(descriptors.numpy())
        th_keypoints = torch.from_numpy(keypoints.numpy())
        args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
        th_descriptors = torch.nn.functional.grid_sample(
            th_descriptors, th_keypoints.view(b, 1, -1, 2), mode="bilinear", **args
        )
        th_descriptors = torch.nn.functional.normalize(
            th_descriptors.reshape(b, c, -1), p=2, dim=1
        )
        return tf.transpose(
            tf.convert_to_tensor(th_descriptors.detach().cpu().numpy()), [0, 2, 1]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--superpoint-weights-file",
        "-S",
        help="pre-trained weights file for SuperPoint model",
        default="superpoint_v1.pth",
    )
    parser.add_argument(
        "--repr-dataset-paths", nargs="+", help="representative dataset", required=True
    )
    parser.add_argument(
        "--nms-radius", type=int, help="Non maximal suppression radius", default=4
    )
    parser.add_argument(
        "--keypoint-threshold",
        default=0.005,
        type=float,
        help="score threshold for keypoint selection",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=1000,
        help="max number of keypoints per image",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="enable quantized tflite model"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output tflite model file path",
        default=Path("superpoint.tflite"),
    )
    args = parser.parse_args()
    config = {
        "nms_radius": args.nms_radius,
        "keypoint_threshold": args.keypoint_threshold,
        "max_keypoints": args.max_keypoints,
    }
    superpoint_tf = SuperPointTF(config)
    superpoint_tf.load_torch_state_dict(args.superpoint_weights_file)
    if args.quantize:
        optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        optimizations = []
    args.output.parent.mkdir(exist_ok=True, parents=True)
    sample_img_path = next(
        chain(
            Path(args.repr_dataset_paths[0]).glob("*.jpg"),
            Path(args.repr_dataset_paths[0]).glob("*.png"),
        )
    )
    img_shape = cv2.imread(str(sample_img_path), cv2.IMREAD_GRAYSCALE).shape
    with args.output.open("wb") as fid:
        tflite_model = superpoint_tf.convert_tflite(
            img_shape,
            repr_dataset_paths=args.repr_dataset_paths,
            optimizations=optimizations,
        )
        fid.write(tflite_model)
    print(f"tflite model saved: {args.output}")
