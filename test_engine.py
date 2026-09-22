import tempfile
import unittest
from pathlib import Path
from engine import Classifier, evaluate

class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.model = Classifier().fit([('invoice payment refund','billing'),('login password account','access')])
    def test_known_prediction(self):
        self.assertEqual(self.model.predict('invoice refund')['label'],'billing')
    def test_abstention(self):
        self.assertIsNone(self.model.predict('telescope')['label'])
        self.assertIsNone(self.model.predict('invoice password',threshold=0.99)['label'])
    def test_no_vocabulary_leakage(self):
        self.model.predict('novelword')
        self.assertNotIn('novelword',self.model.vocabulary)
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'m.json'; self.model.save(path)
            self.assertEqual(self.model.predict('login'),Classifier.load(path).predict('login'))
    def test_metrics_include_abstentions(self):
        result = evaluate(self.model,[('invoice refund','billing'),('unknownword','access')])
        self.assertEqual(result['accuracy'],0.5)
        self.assertEqual(result['coverage'],0.5)
        self.assertLess(result['macro_f1'],1)
    def test_numeric_stability(self):
        result = self.model.predict('invoice '*10000)
        self.assertAlmostEqual(sum(result['probabilities'].values()),1)
