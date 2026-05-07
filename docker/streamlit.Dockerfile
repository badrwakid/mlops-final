FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 10001 appuser

WORKDIR /app

COPY requirements.txt requirements-streamlit.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-streamlit.txt

COPY src ./src
COPY configs ./configs
COPY docs ./docs
COPY monitoring ./monitoring
COPY dvc.yaml dvc.lock README.md ./

RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

