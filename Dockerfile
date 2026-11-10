FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/chroma_db/ /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

ENV HF_HOME=/home/appuser/.cache/huggingface

COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser utils/ ./utils/
COPY --chown=appuser:appuser knowledge_base/nutrition_v1.json ./knowledge_base/nutrition_v1.json
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml

USER appuser

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]