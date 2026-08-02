#!/usr/bin/env bash
#
# Provisiona a infraestrutura de deploy no Google Cloud e configura o GitHub.
#
# Idempotente: cada recurso é criado só se não existir, então rodar de novo é
# seguro. Lê os valores do `.env` (gitignored) — nunca da linha de comando, para
# que o identificador do projeto não fique no histórico do shell.
#
# Uso:
#     cp .env.example .env    # e preencha a seção "provisionamento GCP"
#     ./scripts/setup-gcp.sh
#
# Pré-requisitos: gcloud autenticado (`gcloud auth login`) e gh autenticado
# (`gh auth login`).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"

# ─────────────────────────────────────────────────────────────────────────────
# Saída
# ─────────────────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; OFF=""
fi

step()  { printf '\n%s▸ %s%s\n' "$BOLD" "$1" "$OFF"; }
ok()    { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$1"; }
skip()  { printf '  %s•%s %s (já existia)\n' "$YELLOW" "$OFF" "$1"; }
die()   { printf '\n%serro:%s %s\n' "$RED" "$OFF" "$1" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────
[ -f "$ENV_FILE" ] || die "$ENV_FILE não existe. Comece com: cp .env.example .env"

# `set -a` exporta tudo que o arquivo define; o subshell evita poluir o ambiente
# do chamador com as variáveis EOLICA_ de runtime.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${GCP_PROJECT_ID:?defina GCP_PROJECT_ID no .env}"
: "${GCP_PROJECT_NUMBER:?defina GCP_PROJECT_NUMBER no .env}"
: "${GCP_REGION:?defina GCP_REGION no .env}"
: "${GITHUB_REPO:?defina GITHUB_REPO no .env}"

case "$GCP_PROJECT_ID" in
  seu-projeto-id|"") die "GCP_PROJECT_ID ainda está com o valor de exemplo" ;;
esac

REPOSITORY="eolica"
POOL="github"
PROVIDER="github-provider"
DEPLOY_SA="github-deploy@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

command -v gcloud >/dev/null || die "gcloud não encontrado no PATH"

# `gh` é conveniência, não requisito: ele só grava as três variáveis no GitHub,
# que também dá para preencher pela interface. Abortar o provisionamento inteiro
# do GCP por causa disso seria desproporcional — o script faz o que consegue e
# imprime o resto para copiar.
HAS_GH=1
command -v gh >/dev/null || HAS_GH=0

gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q . \
  || die "gcloud não autenticado. Rode: gcloud auth login"

printf '%sProvisionando%s\n' "$BOLD" "$OFF"
printf '  projeto ..... %s (%s)\n' "$GCP_PROJECT_ID" "$GCP_PROJECT_NUMBER"
printf '  região ...... %s\n' "$GCP_REGION"
printf '  repositório . %s\n' "$GITHUB_REPO"

gcloud config set project "$GCP_PROJECT_ID" --quiet >/dev/null

# Confere que o número bate com o id. Um número errado gera um principalSet
# inválido, e a federação falha só no primeiro deploy — com mensagem opaca.
ACTUAL_NUMBER="$(gcloud projects describe "$GCP_PROJECT_ID" --format='value(projectNumber)')"
[ "$ACTUAL_NUMBER" = "$GCP_PROJECT_NUMBER" ] \
  || die "GCP_PROJECT_NUMBER não bate com o projeto (esperado: ${ACTUAL_NUMBER})"

# ─────────────────────────────────────────────────────────────────────────────
step "APIs"
# ─────────────────────────────────────────────────────────────────────────────
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  --quiet
ok "run, artifactregistry, iamcredentials"

# ─────────────────────────────────────────────────────────────────────────────
step "Artifact Registry"
# ─────────────────────────────────────────────────────────────────────────────
if gcloud artifacts repositories describe "$REPOSITORY" \
     --location="$GCP_REGION" >/dev/null 2>&1; then
  skip "repositório $REPOSITORY"
else
  gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format=docker \
    --location="$GCP_REGION" \
    --description="Imagens da plataforma Eólica SCADA" --quiet
  ok "repositório $REPOSITORY criado"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "Service account do deploy"
# ─────────────────────────────────────────────────────────────────────────────
if gcloud iam service-accounts describe "$DEPLOY_SA" >/dev/null 2>&1; then
  skip "$DEPLOY_SA"
else
  gcloud iam service-accounts create github-deploy \
    --display-name="GitHub Actions — deploy" --quiet
  ok "$DEPLOY_SA criada"
fi

# Permissões mínimas: publicar imagem, administrar o serviço, agir como a SA de
# runtime. Nada além — em particular, nenhum papel de projeto amplo.
for ROLE in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:${DEPLOY_SA}" --role="$ROLE" \
    --condition=None --quiet >/dev/null
  ok "papel $ROLE"
done

# ─────────────────────────────────────────────────────────────────────────────
step "Workload Identity Federation"
# ─────────────────────────────────────────────────────────────────────────────
if gcloud iam workload-identity-pools describe "$POOL" \
     --location=global >/dev/null 2>&1; then
  skip "pool $POOL"
else
  gcloud iam workload-identity-pools create "$POOL" \
    --location=global --display-name="GitHub Actions" --quiet
  ok "pool $POOL criado"
fi

if gcloud iam workload-identity-pools providers describe "$PROVIDER" \
     --location=global --workload-identity-pool="$POOL" >/dev/null 2>&1; then
  skip "provider $PROVIDER"
else
  # A attribute-condition NÃO é opcional. Sem ela, qualquer repositório do
  # GitHub troca seu token OIDC por credencial deste projeto — é a diferença
  # entre federação e porta aberta.
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --location=global \
    --workload-identity-pool="$POOL" \
    --display-name="GitHub OIDC" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository == '${GITHUB_REPO}'" \
    --quiet
  ok "provider $PROVIDER criado, restrito a ${GITHUB_REPO}"
fi

PRINCIPAL="principalSet://iam.googleapis.com/projects/${GCP_PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" \
  --role=roles/iam.workloadIdentityUser \
  --member="$PRINCIPAL" --quiet >/dev/null
ok "somente ${GITHUB_REPO} pode personificar a service account"

# ─────────────────────────────────────────────────────────────────────────────
step "Variáveis no GitHub"
# ─────────────────────────────────────────────────────────────────────────────
PROVIDER_PATH="projects/${GCP_PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"

if [ "$HAS_GH" -eq 1 ]; then
  # Variables, não secrets: nenhuma delas é sigilosa, e variável aparece no log
  # da execução — o que ajuda a depurar em vez de virar `***`.
  gh variable set GCP_PROJECT_ID --repo "$GITHUB_REPO" --body "$GCP_PROJECT_ID"
  gh variable set GCP_DEPLOY_SERVICE_ACCOUNT --repo "$GITHUB_REPO" --body "$DEPLOY_SA"
  gh variable set GCP_WORKLOAD_IDENTITY_PROVIDER --repo "$GITHUB_REPO" --body "$PROVIDER_PATH"
  ok "3 variáveis configuradas em $GITHUB_REPO"
else
  printf '  %s•%s gh não instalado — preencha à mão em:\n' "$YELLOW" "$OFF"
  printf '    https://github.com/%s/settings/variables/actions\n\n' "$GITHUB_REPO"
  printf '    %-32s %s\n' "GCP_PROJECT_ID" "$GCP_PROJECT_ID"
  printf '    %-32s %s\n' "GCP_DEPLOY_SERVICE_ACCOUNT" "$DEPLOY_SA"
  printf '    %-32s %s\n' "GCP_WORKLOAD_IDENTITY_PROVIDER" "$PROVIDER_PATH"
  printf '\n    (para automatizar: winget install GitHub.cli && gh auth login)\n'
fi

# ─────────────────────────────────────────────────────────────────────────────
step "Verificação"
# ─────────────────────────────────────────────────────────────────────────────
gcloud artifacts repositories describe "$REPOSITORY" --location="$GCP_REGION" >/dev/null \
  && ok "Artifact Registry acessível"
gcloud iam workload-identity-pools providers describe "$PROVIDER" \
  --location=global --workload-identity-pool="$POOL" >/dev/null \
  && ok "provider OIDC acessível"
if [ "$HAS_GH" -eq 1 ] \
  && gh variable list --repo "$GITHUB_REPO" 2>/dev/null | grep -q GCP_WORKLOAD_IDENTITY_PROVIDER; then
  ok "variáveis visíveis ao workflow"
  printf '\n%sPronto.%s O próximo push na main dispara o deploy.\n' "$GREEN" "$OFF"
else
  printf '\n%sGCP provisionado.%s Falta só preencher as variáveis no GitHub.\n' "$GREEN" "$OFF"
fi
printf 'Acompanhe em: https://github.com/%s/actions\n\n' "$GITHUB_REPO"
