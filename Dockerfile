# Image that runs the backend on its own Hetzner VPS, where Ansible installs
# it and cron runs it after each trading day's close. See "Deployment" in
# docs/SYSTEM_DESIGN.md and "Running in Docker" in
# docs/INSTALLATION_AND_EXECUTION.md. .dockerignore keeps .env, input/,
# output/, and development files out of the build.

# Python pinned to the same patch version as .python-version
FROM python:3.13.5-slim-bookworm

# uv, pinned to an exact version
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

# Use the image's Python (never a downloaded one), and keep the project's
# environment in /app/.venv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1. The production dependencies, exactly as locked in uv.lock (no dev group).
# Only the files they depend on are copied first, so this layer is rebuilt
# only when the dependencies change.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# 2. The project itself, installed into the same environment with the same
# options
COPY README.md ./
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY src/ ./src/
COPY tests/ ./tests/
RUN uv sync --locked --no-dev

RUN useradd appuser --create-home && mkdir input && chown appuser input && chown -R appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# uv run checks the environment against uv.lock before running, with the
# same --locked --no-dev options as the build, so it uses exactly the
# configured production dependencies. If anything were missing it would
# install it (from the network) rather than run without it; --locked makes
# it fail instead if uv.lock does not match pyproject.toml.
CMD ["uv", "run", "--locked", "--no-dev", "python", "./scripts/run_optimization_and_time_series.py"]
