# gpt-2-repro

Use `eval_wikitext.py/eval_lambada.py` to train or evaluate [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) on one of the supported datasets.

## Quick start

```bash
pip install -r requirements.txt
python eval_wikitext.py/eval_lambada.py --help
```

Or using Docker:

```bash
docker build -t evaluation:latest .
docker run --rm --gpus all evaluation:latest
```
