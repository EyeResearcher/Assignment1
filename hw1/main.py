from easydict import EasyDict
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from basics import run_all_basics_demo
from model import load_data, run, visualize_epochs, visualize_configs
from typing import List, Tuple, Dict, Union
EMBEDDING_TYPES = ["glove-twitter-50", "glove-twitter-100", "glove-twitter-200", "word2vec-google-news-300"]

def _worker(request_path, result_path):
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    stats = run(EasyDict(request['config']), request['dev'], request['train'], request['test'])
    Path(result_path).write_text(json.dumps(stats), encoding='utf-8')


def run_isolated(config, dev_data, train_data, test_data):
    # Every invocation owns a new directory, so previous results cannot be reused.
    with tempfile.TemporaryDirectory(prefix='hw1-') as directory:
        request_path = Path(directory) / 'request.json'
        result_path = Path(directory) / 'result.json'
        request_path.write_text(json.dumps({
            'config': dict(config), 'dev': dev_data,
            'train': train_data, 'test': test_data,
        }), encoding='utf-8')
        completed = subprocess.run([
            sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()),
            '--worker', str(request_path), str(result_path),
        ])
        if completed.returncode:
            code = completed.returncode & 0xffffffff
            raise RuntimeError(
                f"Experiment {config.embeddings} failed with exit code "
                f"{completed.returncode} (0x{code:08X}). See worker output above."
            )
        if not result_path.exists():
            raise RuntimeError(f"Experiment {config.embeddings} produced no results.")
        return json.loads(result_path.read_text(encoding='utf-8'))


def single_run(dev_d: Dict[str, List[Union[str, int]]],
               train_d: Dict[str, List[Union[str, int]]],
               test_d: Dict[str, List[Union[str, int]]]):
    # TODO: once you have completed the model.py, you can run this function to train and evaluate your model, and visualize the training process with a plot
    train_config = EasyDict({
        'batch_size': 64,  # we use batching
        'lr': 0.025,  # learning rate
        'num_epochs': 20,  # the total number of times all the training data is iterated over
        'save_path': 'model.pth',  # path where to save the model
        'embeddings': EMBEDDING_TYPES[0],
        'num_classes': 2,
    })

    epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run_isolated(train_config, dev_d, train_d, test_d)
    visualize_epochs(epoch_train_losses, epoch_dev_loss, "single_run_loss.png")

def explore_embeddings(dev_d: Dict[str, List[Union[str, int]]],
                       train_d: Dict[str, List[Union[str, int]]],
                       test_d: Dict[str, List[Union[str, int]]]):
    # Recompute every experiment from the supplied data. Gensim still caches
    # downloaded embeddings, but old training results are never reused.
    all_accs, all_losses = [], []
    for idx, embedding_type in enumerate(EMBEDDING_TYPES):
        config = EasyDict({
            'batch_size': 64,
            'lr': 0.025,
            'num_epochs': 20,
            'save_path': f'model_{idx}.pth',
            'embeddings': embedding_type,
            'num_classes': 2,
        })
        _, _, losses, accs, _, _ = run_isolated(config, dev_d, train_d, test_d)
        all_accs.append(accs)
        all_losses.append(losses)
    visualize_configs(all_accs, EMBEDDING_TYPES, "Accuracy", "embedding_acc.png")
    visualize_configs(all_losses, EMBEDDING_TYPES, "Loss", "embedding_loss.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assignment 1 sentiment classifier")
    parser.add_argument('--mode', choices=['basics', 'single', 'embeddings', 'all'],
                        default='all')
    parser.add_argument('--worker', nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(*args.worker)
        sys.exit(0)
    if args.mode in ('basics', 'all'):
        run_all_basics_demo()
    if args.mode != 'basics':
        dev_data, train_data, test_data = load_data()
        if args.mode in ('single', 'all'):
            single_run(dev_data, train_data, test_data)
        if args.mode in ('embeddings', 'all'):
            explore_embeddings(dev_data, train_data, test_data)
