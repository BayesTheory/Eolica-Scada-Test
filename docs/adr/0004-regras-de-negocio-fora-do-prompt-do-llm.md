# ADR 0004 — Regra de negócio não mora em prompt de LLM

**Status:** aceito · **Data:** 2026-08-01

## Contexto

O co-piloto da v1 tinha este prompt de sistema:

> **3. REGRA DA POTÊNCIA NEGATIVA:** Se o valor em `previsao_kw` for negativo,
> mostre-o como 0.0 kW. NUNCA mostre um valor de potência negativo.
>
> **4. REGRA DE MANUTENÇÃO:** Se `anomalias_detectadas` for maior que 0 E
> `anomalias_dia_anterior` também for maior que 0, o status da turbina é
> "EM MANUTENÇÃO". Ignore o status "ALERTA" vindo da API nesse caso.
>
> **6. DIAS COM ERRO (BUGADOS):** Se a API retornar um erro interno (status 500)
> para uma data específica (como 2022-02-08), informe ao usuário que os dados
> para esse dia específico estão corrompidos ou indisponíveis.

Três problemas distintos, do menor para o maior:

1. **As regras 3 e 4 são regras de negócio.** Valiam apenas para quem usasse o
   chat. Qualquer outro consumidor da API recebia `-0.3 kW` e o status `ALERTA`,
   sem saber que a organização considerava aquilo `0.0 kW` e `EM MANUTENÇÃO`.
2. **Não eram testáveis.** Verificar a regra 4 exigia chamar a API do Gemini e
   inspecionar texto em linguagem natural.
3. **A regra 6 pedia ao modelo que encobrisse um bug.** O 500 vinha de
   `df.loc["2022-02-08"]` levantando `KeyError` para uma data ausente. Os dados
   não estavam corrompidos; o dia não existia. Em vez de devolver 404,
   instruiu-se o LLM a contar uma história plausível ao operador.

O terceiro é o mais grave: transformou o modelo de linguagem em camada de
ocultação de defeito. Um operador que agisse com base nessa resposta acreditaria
que havia um problema de integridade de dados que nunca existiu.

## Decisão

Regra de negócio vive no domínio. O LLM, se existir, apenas verbaliza um
resultado já decidido.

| Regra da v1 | Onde está agora |
|---|---|
| Potência negativa exibida como zero | `PowerKw.for_display()` — 5 testes |
| Critério de "em manutenção" | `domain/health/services.evaluate_health()` — 5 testes |
| "Explique o 500 ao usuário" | Eliminada. Data ausente é `404 not-found` (RFC 9457) |

O campo `power_kw` da API já vem com clamp; `power_kw_measured` expõe o valor
cru para auditoria. O status `EM_MANUTENCAO` é calculado no servidor e chega
pronto a qualquer cliente.

O co-piloto foi mantido como adaptador opcional (`EOLICA_COPILOT_ENABLED`,
default `false`), sem nenhuma regra no prompt.

## Consequências

**A favor**

- As regras passaram a ser testáveis, versionadas e válidas para todos os
  clientes.
- Trocar de provedor de LLM deixa de ser um risco de regressão de negócio.
- O caminho de erro passou a ter significado: 404 para inexistente, 422 para
  insuficiente, 503 para dependência fora.

**Contra**

- O prompt do co-piloto ficou mais pobre — ele agora relata em vez de decidir.
  É o comportamento desejado.
- Mudar uma regra passou a exigir deploy, e não edição de string. Também é o
  comportamento desejado.

## Princípio geral

Se uma instrução de prompt começa com "se X, então trate como Y", isso é regra
de negócio e pertence ao código. Se pede para explicar um erro de servidor ao
usuário, isso é um bug e pertence ao backlog.
