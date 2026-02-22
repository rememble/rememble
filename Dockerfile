FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY rememble/ rememble/

RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

ENV REMEMBLE_DB_PATH=/data/memory.db

EXPOSE 9909

ENTRYPOINT ["uv", "run", "rememble", "serve"]
