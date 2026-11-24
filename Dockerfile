#Python pinned to the same patch version as .python-version
FROM python:3.13.5-slim-bookworm

#uv pinned to the version the pre-commit hooks use
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

#Use the image's Python
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \ 
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

#Copy in all project dependencies 
COPY pyproject.toml ./
COPY uv.lock ./
COPY .python-version ./
COPY scripts/ ./scripts/
COPY src/ ./src/
COPY README.md ./
COPY tests/ ./tests/
COPY config/ ./config/
RUN uv sync --locked --no-dev --no-install-project

RUN useradd appuser --create-home && mkdir input && chown appuser input && chown -R appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "./scripts/run_optimization_and_time_series.py"]