# Evaluation

Use `eval_wikitext.py/eval_lambada.py` to evaluate [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) on one of the supported datasets.

## Quick start

```bash
pip install -r requirements.txt -r base-requirements.txt
python eval_wikitext.py/eval_lambada.py --help
```

Or using Docker:

```bash
docker build -t evaluation:latest .
docker run --rm --gpus all evaluation:latest eval_wikitext.py/eval_lambada.py --help
```
