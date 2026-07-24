# ADR 0003 — Janelas de leitura são obrigatoriamente contíguas

**Status:** aceito · **Data:** 2026-08-01

## Contexto

O dataset reamostrado tem 65.738 linhas numa grade de 10 minutos — mas a grade
tem buracos:

| Intervalo entre amostras | Ocorrências |
|---|---|
| 10 min (esperado) | 65.653 |
| 20 min | 17 |
| 40 min | 8 |
| 50 min | 3 |
| mais de 24 h | 2 |

São 30 descontinuidades. A v1 montava janelas com `iloc[-n:]` e
`TimeSeriesInferenceDataset`, que fatia um array por posição — sem qualquer
noção de tempo.

Uma janela de 60 passos que atravessa um buraco de 24 horas é apresentada ao
autoencoder como se fossem 10 horas contínuas. O erro de reconstrução dispara, e
o sistema reporta "anomalia na turbina" quando o que houve foi uma anomalia **no
coletor de dados**. É falso positivo com aparência de detecção legítima — o tipo
que faz operador perder confiança e desligar o alarme.

## Decisão

`ReadingWindow` valida contiguidade na construção. Duas portas de entrada:

- `ReadingWindow.of(readings, expected_interval=...)` — **recusa** a sequência
  se houver qualquer salto maior que o intervalo esperado.
- `ReadingWindow.split_on_gaps(readings, expected_interval=..., min_length=...)`
  — **fatia** nos buracos e devolve os trechos íntegros, descartando os curtos
  demais para a janela do modelo.

O caso de uso usa a segunda: um dia com um buraco de duas horas vira dois
segmentos analisados separadamente. As leituras descartadas aparecem em
`DataCoverage.discarded_readings`, e a fração analisada em
`DataCoverage.completeness`.

## Consequências

**A favor**

- Impossível, por construção, entregar uma janela furada a um modelo.
- A recuperação é a correta: não abortar o dia (descartaria dado bom) nem
  ignorar (geraria alarme falso), mas analisar cada trecho íntegro.
- A cobertura vira dado de primeira classe no relatório. Um "OK" com 43,8% de
  cobertura significa muito menos que um "OK" com 100% — e agora o operador
  consegue ver a diferença.

**Contra**

- Dias muito fragmentados podem não produzir nenhum segmento utilizável. É um
  `422`, não um `500`: existe dado, mas não dá para processá-lo como pedido.
- A validação é O(n) na construção. Irrelevante nesta escala.
- `expected_interval` é um parâmetro que precisa acompanhar a grade de
  reamostragem. Vive em `Settings.sampling_interval_minutes`, num lugar só.

## Nota sobre o dado

A escolha de `>` (e não `!=`) na comparação com `expected_interval` é
deliberada: aceita amostras mais próximas que o esperado, recusa buracos. A
grade vem do resample e é exata, mas uma fonte futura com jitter não deveria
quebrar por isso.
