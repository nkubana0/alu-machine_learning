#!/usr/bin/env python3
"""Module to compute style cost in Neural Style Transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""

    def style_cost(self, style_outputs):
        """
        Calculates style cost from style outputs

        Args:
            style_outputs: list of tf.Tensor style outputs

        Returns:
            tf.Tensor: scalar style cost
        """
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != len(self.style_layers):
            raise TypeError("style_outputs must be a list with a length of {}"
                            .format(len(self.style_layers)))

        weights = 1.0 / len(style_outputs)
        cost = 0
        for out, target in zip(style_outputs, self.gram_style_features):
            gm = self.gram_matrix(out)
            cost += weights * tf.reduce_mean(tf.square(gm - target))
        return cost
