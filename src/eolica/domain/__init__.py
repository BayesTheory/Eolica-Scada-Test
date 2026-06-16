"""Camada de domínio: as regras de negócio, e nada além delas.

Restrição arquitetural deste pacote — verificada por
`tests/architecture/test_layer_dependencies.py`:

    nenhum módulo sob `eolica.domain` importa biblioteca de terceiros.

Nem pandas, nem numpy, nem torch, nem pydantic. Só a biblioteca padrão e
`eolica.shared`. O motivo não é purismo: é que as regras que decidem se uma
turbina está saudável precisam ser legíveis e testáveis por quem entende de
turbina, não de tensores. E a suíte de domínio roda em milissegundos.

O preço é ter que converter DataFrame → entidades na fronteira. O ganho é que
uma troca de torch por ONNX, ou de CSV por TimescaleDB, não toca uma linha
daqui.
"""
