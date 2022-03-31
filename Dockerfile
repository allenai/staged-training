FROM ghcr.io/allenai/pytorch:1.9.0-cuda11.1

WORKDIR /stage/

# Install remaining dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace

COPY . .

ENTRYPOINT ["python"]
