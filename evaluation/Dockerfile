# This Dockerfile creates an environment suitable for running `run.py`.
FROM ghcr.io/allenai/pytorch:1.9.0-cuda11.1

WORKDIR /stage/

# Install remaining dependencies.
COPY base-requirements.txt .
COPY requirements.txt .
RUN pip install --no-cache-dir -r base-requirements.txt -r requirements.txt

WORKDIR /run/

COPY tools/ tools
COPY eval_lambada.py .
COPY eval_wikitext.py .

ENTRYPOINT ["python"]
