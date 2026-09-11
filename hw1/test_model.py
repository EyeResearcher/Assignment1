"""Offline regression checks: python -m unittest discover -s hw1."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from easydict import EasyDict
from gensim.models import KeyedVectors
from nltk.tokenize import TreebankWordTokenizer

import main
import model


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # Exercise NLTK word tokenization without needing downloaded Punkt data.
        self.tokenizer = patch.object(model, 'word_tokenize', TreebankWordTokenizer().tokenize)
        self.tokenizer.start()
        self.addCleanup(self.tokenizer.stop)
        self.embeddings = KeyedVectors(vector_size=2)
        self.embeddings.add_vectors(['good', 'bad'], np.eye(2, dtype=np.float32))

    def test_features_and_empty_input(self):
        torch.testing.assert_close(model.featurize('good good bad', self.embeddings),
                                   torch.tensor([2 / 3, 1 / 3]))
        self.assertIsNone(model.featurize('unknown', self.embeddings))
        with self.assertRaisesRegex(ValueError, 'No reviews'):
            model.create_tensor_dataset({'text': ['unknown'], 'label': [0]}, self.embeddings)
        dataset = model.create_tensor_dataset(
            {'text': ['good', 'unknown', 'bad'], 'label': [1, 1, 0]}, self.embeddings)
        self.assertEqual(dataset.tensors[1].tolist(), [1, 0])

    def test_loss_is_sample_weighted_and_eval_disables_gradients(self):
        classifier = model.SentimentClassifier(2, 2)
        with torch.no_grad():
            classifier.linear.weight.copy_(torch.eye(2))
            classifier.linear.bias.zero_()
        x, y = torch.tensor([[8., 0.], [8., 0.], [0., 8.]]), torch.tensor([0, 0, 0])
        loader = model.create_dataloader(torch.utils.data.TensorDataset(x, y), 2, False)
        expected = classifier.loss(classifier(x), y).item()
        grad_states = []
        handle = classifier.register_forward_hook(lambda *args: grad_states.append(torch.is_grad_enabled()))
        loss, acc = model.evaluate(classifier, loader)
        handle.remove()
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertAlmostEqual(acc, 2 / 3)
        self.assertFalse(any(grad_states))
        stats = model.train(classifier, torch.optim.SGD(classifier.parameters(), lr=0),
                            loader, loader, 1)
        self.assertAlmostEqual(stats[0][0], expected, places=6)

    def test_run_trains_saves_reloads_and_is_repeatable(self):
        data = {'text': ['good', 'bad'] * 8, 'label': [0, 1] * 8}
        with tempfile.TemporaryDirectory() as directory:
            config = EasyDict(batch_size=4, lr=.1, num_epochs=6, num_classes=2,
                              embeddings='fake', save_path=str(Path(directory) / 'model.pth'))
            with patch.object(model.gensim.downloader, 'load', return_value=self.embeddings):
                first = model.run(config, data, data, data)
                second = model.run(config, data, data, data)
            self.assertTrue(Path(config.save_path).exists())
            self.assertEqual(first, second)
            self.assertLess(first[0][-1], first[0][0])
            self.assertEqual(first[-1], 1.)

    def test_experiments_always_use_current_data(self):
        data = {'text': ['good'], 'label': [1]}
        with patch.object(main, 'run_isolated', return_value=([], [], [.4], [.8], .4, .8)) as run:
            with patch.object(main, 'visualize_configs'):
                main.explore_embeddings(data, data, data)
                main.explore_embeddings(data, data, data)
        self.assertEqual(run.call_count, 8)
        self.assertEqual([c.args[0].embeddings for c in run.call_args_list[:4]], main.EMBEDDING_TYPES)
        for call in run.call_args_list:
            self.assertIs(call.args[2], data)

    def test_isolated_run_uses_fresh_requests_and_reports_native_crash(self):
        import json
        import subprocess
        config = EasyDict(embeddings='fake')
        requests = []

        def worker(command):
            request_path, result_path = map(Path, command[-2:])
            requests.append((request_path, json.loads(request_path.read_text(encoding='utf-8'))))
            result_path.write_text(json.dumps([[], [], [.4], [.8], .4, .8]))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(main.subprocess, 'run', side_effect=worker):
            for text in ['old', 'new']:
                data = {'text': [text], 'label': [1]}
                result = main.run_isolated(config, data, data, data)
                self.assertEqual(result[-1], .8)
        self.assertNotEqual(requests[0][0], requests[1][0])
        self.assertEqual(requests[1][1]['train']['text'], ['new'])
        self.assertFalse(requests[1][0].exists())
        with patch.object(main.subprocess, 'run', return_value=subprocess.CompletedProcess([], -1073741819)):
            with self.assertRaisesRegex(RuntimeError, '0xC0000005'):
                main.run_isolated(config, {}, {}, {})


if __name__ == '__main__':
    unittest.main()
