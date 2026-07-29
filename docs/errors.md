# Erros da API

Toda resposta de erro segue [RFC 9457 — Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457),
com `Content-Type: application/problem+json`:

```json
{
  "type": "https://github.com/BayesTheory/Eolica-Scada-Test/blob/main/docs/errors.md#not-found",
  "title": "Recurso não encontrado",
  "status": 404,
  "detail": "Telemetria não encontrado para '2030-06-15'",
  "instance": "/api/v1/reports/2030-06-15",
  "code": "not-found",
  "resource": "Telemetria",
  "identifier": "2030-06-15"
}
```

O campo `code` é o contrato estável para tratamento programático — `title` e
`detail` são texto humano e podem mudar de redação. O `type` aponta para a seção
correspondente desta página.

---

## `not-found`

**HTTP 404.** O recurso pedido não existe no acervo.

O caso mais comum é pedir um relatório para uma data fora do período coberto
pela telemetria. `data_range` na resposta de qualquer relatório bem-sucedido
informa o intervalo disponível.

Campos extras: `resource`, `identifier`.

> No v1 este caso devolvia **500**. `df.loc["2022-02-08"]` levanta `KeyError`
> quando a data não está no índice, e a checagem `if df_dia.empty` logo abaixo
> nunca era alcançada. O prompt do co-piloto chegou a instruir o modelo a
> explicar esses 500 ao operador como "dados corrompidos ou indisponíveis" — os
> dados não estavam corrompidos, o dia não existia.

**O que fazer:** verificar a data contra `data_range`. Não repetir a requisição.

---

## `insufficient-data`

**HTTP 422.** O recurso existe, mas não há dado suficiente para processá-lo como
pedido.

Acontece quando um dia existe no acervo mas nenhum trecho contíguo de leituras é
longo o bastante para a janela do modelo. A telemetria tem descontinuidades — um
dia pode ter 40 leituras espalhadas em quatro blocos de 10, e nenhum bloco
comporta uma janela de 60 passos.

Campos extras: `required`, `available`.

**O que fazer:** é um estado legítimo do dado, não uma falha. Repetir não ajuda.
A cobertura de cada dia é visível em `coverage` nos relatórios bem-sucedidos.

---

## `invalid-value`

**HTTP 400.** Um valor viola o domínio permitido.

Grandezas físicas têm faixa validada: vento não é negativo, temperatura não fica
abaixo do zero absoluto, `NaN` não é aceito em lugar nenhum.

**O que fazer:** corrigir a entrada.

---

## `domain-rule-violated`

**HTTP 400.** Uma regra de negócio foi violada — genérico para as invariantes de
domínio que não caem em `invalid-value`.

---

## `model-unavailable`

**HTTP 503.** O modelo não pôde ser carregado do registry.

É transitório e retentável, por isso 503 e não 500.

> No v1 esta situação era silenciada: o `except` do carregador setava
> `self.model = None` e o processo seguia vivo, quebrando só na primeira
> requisição — com stack trace longe da causa. Aqui o serviço falha o readiness
> probe e não recebe tráfego.

**O que fazer:** tentar de novo com backoff. Se persistir, checar
`/health/ready` e o registry.

---

## `data-contract-violated`

**HTTP 500.** Um dado atravessou a fronteira violando o contrato declarado.

É sempre bug: ou o produtor do dado mudou sem avisar, ou o contrato está errado.
O detalhe das violações vai para o log estruturado, nunca para a resposta.

**O que fazer:** abrir incidente. Os dados de entrada precisam ser inspecionados.

---

## `infrastructure-unavailable`

**HTTP 503.** Uma dependência externa (disco, rede, registry) falhou.

**O que fazer:** tentar de novo com backoff.

---

## `internal-error`

**HTTP 500.** Rede de segurança para o que não foi previsto.

O `detail` é genérico de propósito — mensagem de exceção pode conter caminho de
arquivo, host interno ou trecho de query, e nada disso atravessa a fronteira. O
detalhe completo fica no log, correlacionável por `instance` e timestamp.

**O que fazer:** abrir incidente com o `instance` e o horário da requisição.

---

## Erros de validação do framework

**HTTP 422** com o formato do FastAPI (não Problem Details) para entrada
malformada antes de chegar à aplicação — por exemplo `/api/v1/reports/2022-02-31`
ou `/api/v1/reports/nao-e-uma-data`.

A data é tipada como `date` no path, então o framework a valida antes de
qualquer código do projeto rodar. No v1 era query string livre repassada direto
para `.loc` de um DataFrame.
