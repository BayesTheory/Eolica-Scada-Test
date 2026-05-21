import os
import requests
import pandas as pd
import google.generativeai as genai
import json

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
# Lembre-se de colocar sua chave da API do Google
os.environ['GOOGLE_API_KEY'] = "AIzaSyA-LljryCUDVFCD67nQsjZoZwzbgCHtmNQ"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

API_URL = "http://127.0.0.1:8000"

# ==============================================================================
# FERRAMENTAS (FUNÇÕES QUE O LLM PODE CHAMAR)
# ==============================================================================
def gerar_relatorio_diario(data_string: str) -> dict:
    """
    Busca na API de modelos um relatório completo de saúde e previsão para uma data específica.
    """
    print(f"\n[Backend] O LLM chamou esta função para a data: {data_string}")
    
    try:
        response = requests.get(f"{API_URL}/gerar_relatorio_diario/", params={"data_string": data_string})
        response.raise_for_status()
        print("[Backend] -> Resposta da API de modelos recebida com sucesso.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Backend] ERRO: Não foi possível se comunicar com a API de modelos: {e}")
        return {"erro": "O serviço de análise dos especialistas está indisponível."}

# ==============================================================================
# CONFIGURAÇÃO E EXECUÇÃO DO LLM
# ==============================================================================
if __name__ == "__main__":
    
    # --- MUDANÇA PRINCIPAL: Personalidade aprimorada com as novas regras ---
    prompt_personalidade = """
    Você é o "Co-piloto Eólico", um assistente de IA especialista em análise de dados de turbinas eólicas.

    Sua tarefa é responder perguntas do usuário sobre o status de uma turbina, usando a ferramenta `gerar_relatorio_diario`.

    SIGA ESTAS REGRAS ESTRITAMENTE AO GERAR A RESPOSTA:

    1.  **PREÂMBULO:** Comece SEMPRE a resposta com a frase: "Considerando o período de dados de {inicio} até {fim}, aqui está a análise para o dia solicitado:" Use os dados de 'periodo_total_dados' para preencher as datas.

    2.  **FORMATAÇÃO DO RELATÓRIO:** Use Markdown com as seções "Status de Saúde e Anomalias" e "Previsão de Geração de Potência".

    3.  **REGRA DA POTÊNCIA NEGATIVA:** Se o valor em 'previsao_kw' for negativo, mostre-o como 0.0 kW. NUNCA mostre um valor de potência negativo.

    4.  **REGRA DE MANUTENÇÃO:** Se 'anomalias_detectadas' for maior que 0 E 'anomalias_dia_anterior' também for maior que 0, o status da turbina é "EM MANUTENÇÃO". Ignore o status "ALERTA" vindo da API nesse caso.

    5.  **RECOMENDAÇÃO INTELIGENTE:** Termine sempre com uma recomendação concisa e inteligente baseada no contexto. Se o status for "EM MANUTENÇÃO", a recomendação deve ser sobre verificar o progresso da manutenção. Se houver muitas anomalias, recomende ação imediata. Se estiver tudo OK, recomende monitoramento contínuo.

    6.  **DIAS COM ERRO (BUGADOS):** Se a API retornar um erro interno (status 500) para uma data específica (como 2022-02-08), informe ao usuário que os dados para esse dia específico estão corrompidos ou indisponíveis e sugira analisar o dia seguinte.
    """

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=prompt_personalidade,
        tools=[gerar_relatorio_diario]
    )
    
    chat = model.start_chat(enable_automatic_function_calling=True)
    
    print("\n--- Co-piloto Eólico Iniciado (v2.0 - Lógica Aprimorada) ---")
    print("Faça sua pergunta. Ex: 'Qual o status da turbina para o dia 2022-02-07?'")
    print("Digite 'sair' para terminar.")

    while True:
        pergunta_usuario = input("> ")
        if pergunta_usuario.lower() == 'sair':
            break
        
        try:
            response = chat.send_message(pergunta_usuario)
            print("\nCo-piloto Eólico:")
            # Usamos .parts[0].text para garantir que a saída seja sempre texto limpo
            print(response.parts[0].text)

        except Exception as e:
            print(f"\nOcorreu um erro inesperado na comunicação com o LLM: {e}")
        
        print("-" * 20)