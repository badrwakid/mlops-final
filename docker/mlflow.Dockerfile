# docker/mlflow.Dockerfile
FROM python:3.11.9-slim

ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip && pip install mlflow==2.13.0

RUN useradd -m -u 1000 mlflow
WORKDIR /mlflow
RUN chown -R mlflow:mlflow /mlflow

EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
