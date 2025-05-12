#!/usr/bin/env python3
"""NST module: Extracts style and content features from respective images"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class: now extracts features from model outputs"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes NST and extracts style/content features
        """
        tf.enable_eager_execution()

        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.alpha = alpha
        self.beta = beta
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.model = self.load_model()
        self.gram_style_features, self.content_feature = self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that the longest side is 512 px
        Args:
            image: numpy.ndarray - image of shape (h, w, 3)
        Returns:
            Tensor of shape (1, h_new, w_new, 3)
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize_images(
            image, (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC
        )
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        image = tf.expand_dims(image, axis=0)
        return image

    def load_model(self):
        """
        Loads VGG19 model and returns a model that outputs selected style
        and content layers
        Returns:
            tf.keras.Model: model for feature extraction
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [
            vgg.get_layer(name).output
            for name in self.style_layers + [self.content_layer]
        ]

        model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
        return model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates Gram matrix from input layer
        Args:
            input_layer: tf.Tensor of shape (1, h, w, c)
        Returns:
            tf.Tensor of shape (1, c, c)
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        x = tf.reshape(input_layer, shape=(-1, c))
        gram = tf.matmul(x, x, transpose_a=True)
        gram = tf.expand_dims(gram / tf.cast(h * w, tf.float32), axis=0)
        return gram

    def generate_features(self):
        """
        Extracts features from the model and returns Gram matrices and
        content feature
        Returns:
            (gram_style_features, content_feature)
        """
        vgg = tf.keras.applications.vgg19
        style_preprocessed = vgg.preprocess_input(self.style_image * 255)
        content_preprocessed = vgg.preprocess_input(self.content_image * 255)

        outputs_style = self.model(style_preprocessed)
        outputs_content = self.model(content_preprocessed)

        gram_style = [
            self.gram_matrix(output) for output in outputs_style[:-1]
        ]
        content_feature = outputs_content[-1]

        return gram_style, content_feature
