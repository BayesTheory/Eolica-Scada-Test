# ADR 0002 — Feature store com assinatura versionada

**Status:** aceito · **Data:** 2026-08-01

## Contexto

Na v1, as features de lag do modelo de previsão eram construídas em dois lugares:

```python
# train_models.py — treino
df_lags[f"{col}_lag_{lag}"] = df_input[col].shift(lag)

# forecaster.py — serving
feature_vector[f"{col}_lag_{lag}"] = [last_window[col].iloc[-lag]]
```

Duas implementações da mesma ideia, em arquivos diferentes, sem nenhum teste
ligando uma à outra.

Pior: o número de lags vinha de caminhos distintos da configuração —
`model_params['params']['n_lags']` no treino, `forecasting_params['n_lags']` no
serving. **Nenhuma das duas chaves existia no `config.yaml`.** Ambas caíam no
mesmo default `6`, e o sistema funcionava por coincidência.

Bastava alguém adicionar `n_lags: 12` na seção que parecia certa para o modelo
passar a receber, em produção, features com significado diferente das do treino.
Sem erro, sem log, sem exceção: apenas previsão pior. É a classe de bug mais
cara que existe em ML, porque não se manifesta como falha.

## Decisão

Uma única classe, `LagFeatureView`, define e materializa as features:

- `build_training_matrix()` e `build_inference_vector()` compartilham a função
  privada `_lag_column()`. Não existe outra forma de construir um lag no
  projeto.
- A ordem das colunas é **derivada** (`sorted(features)` × lag crescente), nunca
  informada. Duas instâncias com as mesmas features em ordem diferente produzem
  a mesma ordem de colunas.
- A view expõe uma `signature` — hash de `(features, target, n_lags)` — para ser
  gravada junto do modelo no registry e comparada no carregamento.

E um teste que é o ponto inteiro do módulo:

```python
def test_treino_e_serving_produzem_o_mesmo_vetor():
    training_row = features.loc[target_time]
    serving_row = view.build_inference_vector(history_up_to_t_minus_1).iloc[0]
    pd.testing.assert_series_equal(training_row, serving_row, check_dtype=True)
```

## Consequências

**A favor**

- Reintroduzir uma segunda implementação de lag quebra o CI.
- Mudar `n_lags` muda a assinatura; um modelo treinado com a antiga recusa a
  servir com a nova. O bug silencioso vira erro alto no readiness probe.
- A ordem das colunas deixa de depender de `model.feature_names_in_`, que é uma
  rede de segurança do scikit-learn que um export para ONNX ou um `Booster` cru
  não teriam.

**Contra**

- Não é um feature store de verdade: sem materialização offline, sem
  point-in-time correctness para múltiplas entidades, sem serving de baixa
  latência. Resolve o problema deste sistema e não mais que isso.
- Introduzir features com janelas móveis (`_std_1h`, `_roc_1h`, que a análise
  exploratória da v1 indicou como promissoras) exige estender a view — hoje ela
  só sabe fazer lag.

## Alternativas consideradas

- **Feast.** Rejeitada: traz Redis, registry próprio e um modelo de deployment
  desproporcional para uma turbina e cinco features.
- **Só documentar que os dois lugares precisam concordar.** É o que a v1 fazia,
  implicitamente. Documentação não falha o build.
