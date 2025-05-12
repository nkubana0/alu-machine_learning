#!/usr/bin/env python3
"""NST module: Initializes a neural style transfer instance"""

import numpy as np
import tensorflow as tf


class NST:
    """Class for Neural Style Transfer initialization"""

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
        Class constructor for NST
        Args:
            style_image: numpy.ndarray - style reference image (h, w, 3)
            content_image: numpy.ndarray - content reference image (h, w, 3)
            alpha: float - weight for content cost
            beta: float - weight for style cost
        """
        tf.enable_eager_execution()

        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")

        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.alpha = alpha
        self.beta = beta
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

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
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize_images(image, (new_h, new_w),
                                       method=tf.image.ResizeMethod.BICUBIC)
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        image = tf.expand_dims(image, axis=0)
        return image