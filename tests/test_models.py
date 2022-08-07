import unittest
from ssl_study.models import SimpleSupervisedModel
from configs import get_config

args = get_config()

class BaselineModelTest(unittest.TestCase):
    def setUp(self):
        self.baseline = SimpleSupervisedModel(args)

    def test_baseline_output_shape(self):
        model = self.baseline.get_model()
        output_shape = model.layers[-1].output_shape
        self.assertEqual(output_shape, (None, 200))