#!/usr/bin/env python3
"""NST module: Computes total style cost from all style layers"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class with style cost computation"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializes NST and extracts features"""
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
        """Scales image to have longest side of 512 pixels"""
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
        """Loads VGG19 model and extracts specified layer outputs"""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [
            vgg.get_layer(name).output
            for name in self.style_layers + [self.content_layer]
        ]
        return tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Computes the Gram matrix for a given layer"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        x = tf.reshape(input_layer, shape=(-1, c))
        gram = tf.matmul(x, x, transpose_a=True)
        return tf.expand_dims(gram / tf.cast(h * w, tf.float32), axis=0)

    def generate_features(self):
        """Extracts features for style and content comparison"""
        vgg = tf.keras.applications.vgg19
        style_input = vgg.preprocess_input(self.style_image * 255)
        content_input = vgg.preprocess_input(self.content_image * 255)

        outputs_style = self.model(style_input)
        outputs_content = self.model(content_input)

        gram_style = [self.gram_matrix(o) for o in outputs_style[:-1]]
        content_feat = outputs_content[-1]
        return gram_style, content_feat

    def layer_style_cost(self, style_output, gram_target):
        """Computes cost for a single style layer"""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        _, h, w, c = style_output.shape
        if gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(c, c)
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the overall style cost
        Args:
            style_outputs: list of tf.Tensor outputs from model
        Returns:
            Total style cost (scalar tensor)
        """
        if not isinstance(style_outputs, list) or \
                len(style_outputs) != len(self.style_layers):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len(self.style_layers)
                )
            )

        weights = 1.0 / len(style_outputs)
        total_cost = 0

        for output, target in zip(style_outputs, self.gram_style_features):
            total_cost += self.layer_style_cost(output, target) * weights

        return total_cost
