import unittest

import numpy as np
import numpy.testing as npt

import tensorflow as tf

from IQT import models

tf.config.set_visible_devices([], 'GPU')  # This is a test pipeline, no need for GPU util


class ModelConfigCase(unittest.TestCase):

    def test_loaders(self):

        with self.assertRaises(ValueError):
            models.config_model("non-existent model", 16, train_preprocessors=[])

        for conf in ["UNet-T1", "UNet-PreFusion", "UNet"]:
            model = models.config_model(conf, 16, train_preprocessors=[])

            self.assertEqual(model.output_shape[1:], (16, 16, 16, 6))


if __name__ == '__main__':
    unittest.main()
