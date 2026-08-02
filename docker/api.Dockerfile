# syntax=docker/dockerfile:1.7
#
# Imagem única servindo API e frontend. Multi-stage para que nem o compilador
# Python nem o Node cheguem à imagem final, e usuário não-root porque um
# processo que lê CSV e serve HTTP não tem razão para ser root.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — build do frontend
# ─────────────────────────────────────────────────────────────────────────────
FROM node:22-slim AS frontend

WORKDIR /build

# `npm ci` a partir do lockfile antes de copiar o código: a camada de
# dependências é reaproveitada enquanto o lockfile não muda.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# `tsc --noEmit` roda junto: um contrato quebrado entre front e API derruba o
# build da imagem, não o navegador do operador.
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — dependências Python
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/eolica/__init__.py src/eolica/__init__.py

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

COPY src/ src/
RUN /opt/venv/bin/pip install --no-deps .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

# Caminhos explícitos, e não herdados da descoberta de raiz do projeto. Num
# pacote instalado o módulo vive em site-packages, longe dos dados — e confiar
# em heurística de caminho foi exatamente o que fez o container subir e morrer
# no lifespan procurando dados em `/opt/venv/lib/python3.11/data/`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    EOLICA_ENVIRONMENT=production \
    EOLICA_LOG_FORMAT=json \
    EOLICA_DATA_PATH=/app/data/processed/scada_resampled_10min_base.csv \
    EOLICA_SAMPLE_DATA_PATH=/app/data/samples/scada_sample.csv \
    EOLICA_FRONTEND_DIR=/app/frontend/dist \
    PORT=8080

RUN groupadd --system --gid 1001 eolica \
    && useradd --system --uid 1001 --gid eolica --create-home eolica

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=frontend --chown=eolica:eolica /build/dist/ frontend/dist/
COPY --chown=eolica:eolica data/samples/ data/samples/
COPY --chown=eolica:eolica data/metadata/ data/metadata/

USER eolica

# 8080 é o default do Cloud Run; `$PORT` sobrescreve em qualquer plataforma.
EXPOSE 8080

# O probe bate em /health/live, que não toca modelo nem disco: uma dependência
# lenta não pode fazer o orquestrador matar um processo saudável.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8080')}/health/live\", timeout=2).status==200 else 1)"

# `sh -c` para que `$PORT` seja expandido. Sem isso o Cloud Run injeta a porta e
# o processo escuta noutra, e o deploy falha no health check com uma mensagem
# que não diz nada sobre porta.
CMD ["sh", "-c", "exec uvicorn eolica.interfaces.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8080} --no-access-log"]
