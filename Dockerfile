FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/pathanchor/src

WORKDIR /opt/pathanchor

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir -e .

CMD ["pathanchor-verify", "--skip-tools"]
