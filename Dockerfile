FROM ghcr.io/mlflow/mlflow:latest

USER root

# Install psycopg2 dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    && pip install psycopg2-binary \
    && apt-get clean

RUN useradd -m mlflow

USER mlflow