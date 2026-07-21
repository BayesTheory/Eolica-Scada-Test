# syntax=docker/dockerfile:1.7
#
# Imagem da API. Multi-stage para que compilador e cache de build não cheguem
# à imagem final, e usuário não-root porque um processo que só lê CSV e serve
# HTTP não tem razão nenhuma para ser root dentro do container.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — dependências
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copiar só os metadados antes do código faz o Docker reaproveitar a camada de
# dependências enquanto o pyproject não muda — o que é quase sempre.
COPY pyproject.toml README.md ./
COPY src/eolica/__init__.py src/eolica/__init__.py

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

COPY src/ src/
RUN /opt/venv/bin/pip install --no-deps .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    EOLICA_ENVIRONMENT=production \
    EOLICA_LOG_FORMAT=json

RUN groupadd --system --gid 1001 eolica \
    && useradd --system --uid 1001 --gid eolica --create-home eolica

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=eolica:eolica data/samples/ data/samples/
COPY --chown=eolica:eolica data/metadata/ data/metadata/

USER eolica

EXPOSE 8000

# O probe bate em /health/live, que não toca modelo nem disco: uma dependência
# lenta não pode fazer o orquestrador matar um processo saudável.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).status==200 else 1)"

ENTRYPOINT ["uvicorn"]
CMD ["eolica.interfaces.api.app:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--no-access-log"]
