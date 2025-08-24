# Salve este código como analise_final.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================================================================
# CONFIGURAÇÕES GERAIS
# ==============================================================================
MLFLOW_TRACKING_URI = "http://localhost:5000"
REGISTERED_MODEL_NAME = "wind-turbine-health-specialist"
DATA_FILE_PATH = r"C:\Users\riana\OneDrive\Desktop\scada wind\PontoUnico\Data\scada_resampled_10min_base.csv"
FEATURES_TO_USE = ['WindSpeed', 'PowerOutput', 'PitchDeg', 'GeneratorTemperature', 'RotorSpeed']
INPUT_WINDOW_STEPS = 60
STATUS_NORMAL = 10
STATUS_FALHA = 13
HORAS_ANTECEDENCIA_ALERTA = 2 # Quantas horas antes de uma falha vamos procurar por sinais

# ==============================================================================
# LÓGICA DE PROCESSAMENTO E CLASSES (AUTOSSUFICIENTE)
# ==============================================================================
def process_scada_data(df):
    """Cria a coluna 'Status_rounded' a partir de 'StatusAnlage'."""
    if 'StatusAnlage' in df.columns:
        df['Status_rounded'] = df['StatusAnlage'].round()
        print("-> Coluna 'Status_rounded' criada com sucesso em memória.")
    else:
        print("ERRO CRÍTICO: A coluna 'StatusAnlage' é necessária mas não foi encontrada no CSV.")
        exit()
    return df

class TimeSeriesInferenceDataset(Dataset):
    """Dataset customizado para inferência eficiente."""
    def __init__(self, data, window_size):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.num_samples = len(self.data) - self.window_size + 1
    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.data[idx : idx + self.window_size]

def load_model_and_scaler(model_name):
    """Carrega o modelo e o scaler do MLflow."""
    print(f"Carregando modelo '{model_name}'...")
    try:
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.pytorch.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(name=model_name, stages=["None"])[0]
        run_id = model_version.run_id
        local_dir = "mlflow_artifacts_final"
        if not os.path.exists(local_dir): os.makedirs(local_dir)
        local_path = client.download_artifacts(run_id, ".", local_dir)
        scaler_path = os.path.join(local_path, "health_model_scaler.pkl")
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
        print("-> Modelo e scaler carregados com sucesso.")
        return model, scaler
    except Exception as e:
        print(f"\nERRO CRÍTICO ao carregar o modelo: {e}")
        return None, None

# ==============================================================================
# LÓGICA PRINCIPAL DA ANÁLISE
# ==============================================================================
if __name__ == "__main__":
    print("--- INICIANDO ANÁLISE FINAL DO DETECTOR DE ANOMALIAS ---")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model, scaler = load_model_and_scaler(REGISTERED_MODEL_NAME)

    if model and scaler:
        # 1. CARREGAR E PROCESSAR DADOS
        print(f"\n[ETAPA 1/5] Carregando e processando dados de '{os.path.basename(DATA_FILE_PATH)}'...")
        df_raw = pd.read_csv(DATA_FILE_PATH, index_col='Datetime', parse_dates=True)
        df_full = process_scada_data(df_raw).dropna(subset=FEATURES_TO_USE)

        # 2. DEFINIR LIMIAR DE ANOMALIA
        print("\n[ETAPA 2/5] Definindo limiar de anomalia com base nos dados normais...")
        df_normal = df_full[df_full['Status_rounded'] == STATUS_NORMAL]
        normal_data_scaled = scaler.transform(df_normal[FEATURES_TO_USE])
        normal_dataset = TimeSeriesInferenceDataset(normal_data_scaled, INPUT_WINDOW_STEPS)
        normal_loader = DataLoader(normal_dataset, batch_size=256, shuffle=False)
        
        errors_normal_list = []
        with torch.no_grad():
            for seq_batch in normal_loader:
                reconstructed = model(seq_batch)
                error = torch.mean((seq_batch - reconstructed)**2, dim=(1, 2))
                errors_normal_list.append(error.cpu().numpy())
        errors_normal = np.concatenate(errors_normal_list)
        threshold = np.percentile(errors_normal, 99.5)
        print(f"-> LIMIAR DE ANOMALIA DEFINIDO: {threshold:.6f}")

        # 3. AVALIAÇÃO COMPLETA E MATRIZ DE CONFUSÃO
        print("\n[ETAPA 3/5] Gerando Matriz de Confusão...")
        df_eval = df_full[df_full['Status_rounded'].isin([STATUS_NORMAL, STATUS_FALHA])].copy()
        eval_data_scaled = scaler.transform(df_eval[FEATURES_TO_USE])
        eval_dataset = TimeSeriesInferenceDataset(eval_data_scaled, INPUT_WINDOW_STEPS)
        eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)
        
        errors_eval_list = []
        with torch.no_grad():
            for seq_batch in eval_loader:
                reconstructed = model(seq_batch)
                error = torch.mean((seq_batch - reconstructed)**2, dim=(1, 2))
                errors_eval_list.append(error.cpu().numpy())
        errors_eval = np.concatenate(errors_eval_list)
        
        true_labels = df_eval['Status_rounded'].iloc[INPUT_WINDOW_STEPS-1:].map({STATUS_NORMAL: 0, STATUS_FALHA: 1}).to_numpy()
        predicted_labels = (errors_eval > threshold).astype(int)
        
        cm = confusion_matrix(true_labels, predicted_labels)
        vn, fp, fn, vp = cm.ravel()
        print(f"-> Verdadeiros Negativos (VN): {vn}")
        print(f"-> Falsos Positivos (FP):    {fp} (Alarmes Falsos)")
        print(f"-> Falsos Negativos (FN):    {fn} (Falhas Perdidas)")
        print(f"-> Verdadeiros Positivos (VP): {vp}")
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomalia'])
        fig, ax = plt.subplots(figsize=(8, 8)); disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title('Matriz de Confusão do Detector de Anomalias', fontsize=16)
        plt.savefig('matriz_confusao.png'); plt.close()
        print("-> Gráfico 'matriz_confusao.png' salvo.")

        # 4. CAÇA A SINAIS PRECOCES
        print("\n[ETAPA 4/5] Investigando sinais de alerta precoce...")
        errors_df = pd.DataFrame(data=errors_eval, index=df_eval.index[INPUT_WINDOW_STEPS-1:], columns=['ReconstructionError'])
        df_analysis = df_eval.join(errors_df)
        df_analysis['status_anterior'] = df_analysis['Status_rounded'].shift(1)
        transicoes = df_analysis[(df_analysis['status_anterior'] == STATUS_NORMAL) & (df_analysis['Status_rounded'] == STATUS_FALHA)]
        
        best_example = None
        if not transicoes.empty:
            print(f"-> Encontradas {len(transicoes)} transições de Normal para Falha.")
            for fault_timestamp in transicoes.index:
                search_end_time = fault_timestamp
                search_start_time = fault_timestamp - pd.Timedelta(hours=HORAS_ANTECEDENCIA_ALERTA)
                pre_fault_window = df_analysis.loc[search_start_time:search_end_time]
                early_signals = pre_fault_window[pre_fault_window['ReconstructionError'] > threshold]
                if not early_signals.empty:
                    first_signal_timestamp = early_signals.index[0]
                    best_example = {"fault_start": fault_timestamp, "signal_time": first_signal_timestamp}
                    print(f"-> SUCESSO! Encontrado alerta precoce em {first_signal_timestamp} para a falha de {fault_timestamp}.")
                    break
        
        # 5. VISUALIZAÇÃO E EXPORTAÇÃO
        print("\n[ETAPA 5/5] Gerando relatórios e gráficos finais...")
        if best_example:
            # PLOT 1: ALERTA PRECOCE
            fault_start, signal_time = best_example['fault_start'], best_example['signal_time']
            plot_start = signal_time - pd.Timedelta(hours=1); plot_end = fault_start + pd.Timedelta(hours=1)
            plot_data = df_analysis.loc[plot_start:plot_end]
            fig, ax1 = plt.subplots(figsize=(20, 8)); ax2 = ax1.twinx()
            ax1.plot(plot_data.index, plot_data['PowerOutput'], color='blue', label='PowerOutput Real')
            ax2.plot(plot_data.index, plot_data['ReconstructionError'], color='orange', label='Erro de Reconstrução (MSE)')
            ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Limiar de Anomalia ({threshold:.4f})')
            ax2.axvline(x=signal_time, color='green', linestyle=':', linewidth=3, label=f"Alerta Precoce ({signal_time.time()})")
            ax2.axvline(x=fault_start, color='black', linestyle=':', linewidth=3, label=f"Início da Falha Oficial ({fault_start.time()})")
            ax1.set_ylabel('PowerOutput (kW)', color='blue'); ax2.set_ylabel('Erro de Reconstrução', color='orange')
            fig.suptitle('Detecção de Anomalia Preditiva - Alerta Precoce', fontsize=18); fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.savefig('grafico_alerta_precoce.png'); plt.close()
            print("-> Gráfico 'grafico_alerta_precoce.png' salvo.")
            
            # PLOT 2: DIAGNÓSTICO DETALHADO
            window_start_time = signal_time - pd.Timedelta(minutes=(INPUT_WINDOW_STEPS - 1) * 10)
            window_df = df_full.loc[window_start_time:signal_time]
            real_sequence_scaled = scaler.transform(window_df[FEATURES_TO_USE])
            input_tensor = torch.tensor(real_sequence_scaled, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad(): reconstructed_sequence_scaled = model(input_tensor)
            df_real = pd.DataFrame(scaler.inverse_transform(real_sequence_scaled), index=window_df.index, columns=FEATURES_TO_USE)
            df_reconstructed = pd.DataFrame(scaler.inverse_transform(reconstructed_sequence_scaled.squeeze(0).cpu().numpy()), index=window_df.index, columns=FEATURES_TO_USE)
            
            fig, axes = plt.subplots(nrows=len(FEATURES_TO_USE), ncols=1, figsize=(15, 12), sharex=True)
            fig.suptitle(f"Análise da Reconstrução no Momento do Alerta ({signal_time})", fontsize=18)
            for i, feature in enumerate(FEATURES_TO_USE):
                axes[i].plot(df_real.index, df_real[feature], color='blue', label='Sinal Real')
                axes[i].plot(df_reconstructed.index, df_reconstructed[feature], color='red', linestyle='--', label='Sinal Reconstruído')
                axes[i].axvline(x=signal_time, color='green', linestyle=':', linewidth=2, label='Momento do Alerta')
                axes[i].set_ylabel(feature); axes[i].legend(loc='upper left'); axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig('grafico_analise_anomalia.png'); plt.close()
            print("-> Gráfico 'grafico_analise_anomalia.png' salvo.")
        else:
            print("-> Nenhum sinal de alerta claro foi encontrado nos períodos de pré-falha. Gráficos de alerta não gerados.")
            
        # EXPORTAR RELATÓRIO CSV
        df_falha_analysis = df_analysis[df_analysis['Status_rounded'] == STATUS_FALHA].copy()
        anomalies_report = df_falha_analysis[df_falha_analysis['ReconstructionError'] > threshold]
        anomalies_report.sort_values(by='ReconstructionError', ascending=False, inplace=True)
        anomalies_report.to_csv('relatorio_anomalias.csv')
        print("-> Relatório 'relatorio_anomalias.csv' com anomalias de falha detectadas foi salvo.")

        print("\n--- ANÁLISE FINAL CONCLUÍDA ---")