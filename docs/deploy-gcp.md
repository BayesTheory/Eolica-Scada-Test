# Deploy no Google Cloud Run

O serviço é uma imagem única servindo API e frontend na mesma origem. Cloud Run
com scale-to-zero: parado, custa nada.

## Por que Cloud Run

O serviço é stateless, lê CSV e responde HTTP. GKE traria um plano de controle
para gerenciar; uma VM traria patching de SO e um processo para supervisionar.
Cloud Run entrega HTTPS, escalonamento e revisões com rollback sem nada disso.

## Autenticação: sem chave, sempre

O workflow usa **Workload Identity Federation**. O GitHub apresenta seu token
OIDC, o GCP o troca por uma credencial de curta duração, e nenhum segredo de
longa duração existe em lugar nenhum.

Não use chave JSON de service account nos secrets do GitHub. Este repositório já
teve uma chave de API commitada em texto claro — fechar o ciclo reintroduzindo
uma credencial permanente seria trocar um erro por outro.

## Preparação — comandos que você roda uma vez

Exigem suas credenciais, então precisam ser executados por você.

```bash
# ── ajuste estas quatro linhas ───────────────────────────────────────────────
export PROJECT_ID="seu-projeto-id"
export PROJECT_NUMBER="000000000000"           # gcloud projects describe $PROJECT_ID --format='value(projectNumber)'
export GITHUB_REPO="SEU-USUARIO/SEU-REPO"
export REGION="southamerica-east1"             # São Paulo

gcloud config set project "$PROJECT_ID"
```

### 1. Habilitar as APIs

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  cloudbuild.googleapis.com
```

### 2. Repositório de imagens

```bash
gcloud artifacts repositories create eolica \
  --repository-format=docker \
  --location="$REGION" \
  --description="Imagens da plataforma Eólica SCADA"
```

### 3. Service account do deploy

Recebe só o que precisa: publicar imagem, administrar o serviço Cloud Run, e
agir como a service account de runtime. Nada além.

```bash
gcloud iam service-accounts create github-deploy \
  --display-name="GitHub Actions — deploy"

DEPLOY_SA="github-deploy@${PROJECT_ID}.iam.gserviceaccount.com"

for ROLE in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${DEPLOY_SA}" --role="$ROLE"
done
```

### 4. Workload Identity Federation

```bash
gcloud iam workload-identity-pools create github \
  --location=global --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global \
  --workload-identity-pool=github \
  --display-name="GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == '${GITHUB_REPO}'"
```

> A `--attribute-condition` não é opcional. Sem ela, **qualquer repositório do
> GitHub** pode trocar seu token por credencial do seu projeto. É a diferença
> entre federação e porta aberta.

Autorizar só este repositório a personificar a service account:

```bash
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${GITHUB_REPO}"
```

### 5. Variáveis no GitHub

Em **Settings → Secrets and variables → Actions → Variables** (variáveis, não
secrets — nenhuma delas é sigilosa):

| Nome | Valor |
|---|---|
| `GCP_PROJECT_ID` | o seu `$PROJECT_ID` |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | `github-deploy@$PROJECT_ID.iam.gserviceaccount.com` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | veja abaixo |

> Identificador de projeto não é segredo, mas também não precisa ir para um
> repositório público: ele torna o projeto enumerável e dá a quem quiser tentar
> abuso um alvo nomeado. Por isso ele vive em *variável do GitHub Actions*, e a
> documentação usa placeholder.

```bash
echo "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/providers/github-provider"
```

Crie também o environment `production` em **Settings → Environments**. Com uma
regra de revisão obrigatória, todo deploy passa a exigir aprovação humana.

## Deploy

Push na `main` dispara. Ou manualmente pela aba Actions.

O workflow builda, publica pelo **digest do SHA** (nunca `latest`), faz deploy,
roda smoke test contra a revisão publicada e **devolve o tráfego para a revisão
anterior se o smoke test falhar**.

O smoke test verifica três coisas: o readiness responde pronto, uma data
inexistente devolve 404 — regressão do bug central da v1 — e a SPA é servida na
raiz.

## Configuração de runtime

| Flag | Valor | Por quê |
|---|---|---|
| `--min-instances=0` | scale-to-zero | Portfólio não tem tráfego constante; o custo cai a zero parado |
| `--cpu-boost` | ligado | Compensa a calibração do detector no boot, que roda no lifespan |
| `--concurrency=40` | | A API é I/O-bound; 40 requisições por instância é folgado |
| `--max-instances=3` | | Teto de custo. Sem ele, um scraper vira uma fatura |
| `--memory=1Gi` | | O sample cabe com folga; o dataset completo em memória pede ~512Mi |

### O trade-off do scale-to-zero

Com `min-instances=0`, a primeira requisição depois de ociosidade paga o cold
start **mais** a calibração do detector. Com o sample versionado são poucos
segundos. Com o dataset completo, mais.

Se a latência da primeira visita incomodar, `--min-instances=1` resolve — e passa
a custar, porque uma instância fica sempre viva. Para um portfólio, o cold start
é o trade certo.

## Custo

| Item | Estimativa |
|---|---|
| Cloud Run | **R$ 0** — 2M requisições e 360k GiB-s por mês no free tier |
| Artifact Registry | ~US$ 0,05/mês (imagem ~500 MB, 0,5 GB no free tier) |
| Egress | desprezível |

Coloque um orçamento com alerta assim mesmo:

```bash
gcloud billing budgets create \
  --billing-account="$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingAccountName)' | cut -d/ -f2)" \
  --display-name="eolica-scada" \
  --budget-amount=20BRL \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9
```

## Dados em produção

A imagem embute apenas o **sample versionado** (duas semanas). O container sobe
com ele e registra um `WARNING` explícito dizendo que não é o dataset completo.

Para servir o dataset inteiro, monte um bucket:

```bash
gcloud storage buckets create "gs://${PROJECT_ID}-eolica-data" --location="$REGION"
gcloud storage cp data/processed/scada_resampled_10min_base.csv "gs://${PROJECT_ID}-eolica-data/"
```

E adicione ao deploy:

```
--add-volume=name=data,type=cloud-storage,bucket=${PROJECT_ID}-eolica-data
--add-volume-mount=volume=data,mount-path=/mnt/data
--set-env-vars=EOLICA_DATA_PATH=/mnt/data/scada_resampled_10min_base.csv
```

## Rollback manual

```bash
gcloud run revisions list --service=eolica-scada --region="$REGION"
gcloud run services update-traffic eolica-scada \
  --region="$REGION" --to-revisions=eolica-scada-00007-abc=100
```

## O que este deploy não faz

Explicitamente, para não parecer que faz:

- **Sem autenticação.** `--allow-unauthenticated` deixa a API pública. É
  adequado para uma demo e inadequado para dado operacional real.
- **Sem rate limiting.** Um cliente insistente consome sua cota. O
  `--max-instances=3` limita o custo, não o abuso.
- **Sem WAF nem Cloud Armor.**
- **Sem domínio próprio.** A URL é a gerada pelo Cloud Run.
- **Sem tracing distribuído.** Há métricas e log estruturado, não spans.
