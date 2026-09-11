# EN.705.641.81 — Assignment 1

Use the `ssw_hw1` Conda environment (Python 3.10.13). The existing `.venv`
uses Python 3.13 and does not contain the assignment dependencies. Select
`ssw_hw1` as the Python interpreter in your editor too.

From this `Assignment1` directory:

```powershell
conda activate ssw_hw1
python hw1/main.py --mode basics
python hw1/main.py --mode single
python hw1/main.py --mode embeddings
```

Omit `--mode` to run all three stages. Models and plots are written to the
current directory. The embedding comparison trains all four configurations
anew; old `imdb_data.pkl` and `emb_results_*.pkl` files are no longer used.
Each training experiment runs in a fresh Python subprocess, including the
single-run exercise. Temporary input and result files belong only to that
invocation and are cleaned up afterward. Native crashes print their exit
code and stop the comparison instead of producing incomplete plots.
Gensim still reuses its downloaded embeddings. The larger embeddings require
substantial RAM even without copying their entire vocabulary.

For a fresh environment, follow `hw1/hw1.md`, then download the tokenizer once:

```powershell
python -m nltk.downloader punkt
```

This command matches the assignment's pinned NLTK 3.8.1. Imports no longer
download resources automatically. The existing `ssw_hw1` environment already
has the tokenizer installed.

Run the offline regression checks:

```powershell
python -m unittest discover -s hw1
```

These checks use tiny in-memory embeddings; they do not download IMDB or
the pretrained embeddings. Regenerate the plots before using them in the
submission: existing plots predate the loss and tokenization fixes.
