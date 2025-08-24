"""
Script de Geração de Relatório Comparativo (v1.1)

Conecta-se ao MLflow, busca os resultados dos principais modelos treinados,
e gera uma tabela e um gráfico de barras comparando suas métricas de desempenho.
"""
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configurações ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Wind Power Forecasting"

def generate_comparison_report():
    """Função principal para ser chamada pelo main.py."""
    print("--- Gerando Relatório Comparativo de Modelos ---")
    
    try:
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            print(f"ERRO: Experimento '{EXPERIMENT_NAME}' não encontrado.")
            return
            
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.'mlflow.parentRunId' IS NULL AND metrics.avg_rmse_kW > 0"
        )
        if runs_df.empty:
            print("Nenhum run de CV completo (com previsão de energia) encontrado para comparar.")
            return
    except Exception as e:
        print(f"ERRO ao conectar ao MLflow ou buscar dados: {e}")
        print("Certifique-se de que o servidor MLflow está rodando (`python -m mlflow ui`)")
        return

    metric_cols = ['metrics.avg_rmse_kW', 'metrics.avg_mae_kW', 'metrics.avg_r2_score']
    tag_col = 'tags.model_key'
    
    report_df = runs_df[[tag_col] + metric_cols].copy()
    report_df.rename(columns={
        tag_col: 'Modelo',
        'metrics.avg_rmse_kW': 'RMSE (kW)',
        'metrics.avg_mae_kW': 'MAE (kW)',
        'metrics.avg_r2_score': 'R² Score'
    }, inplace=True)

    report_df = report_df.loc[report_df.groupby('Modelo')['RMSE (kW)'].idxmin()]
    report_df.set_index('Modelo', inplace=True)
    report_df = report_df.sort_values('RMSE (kW)')

    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, 'comparison_report.csv')
    report_df.to_csv(table_path)
    print("\nTabela Comparativa de Métricas (Melhor Run de Cada Modelo):")
    print(report_df.round(3))
    print(f"\nTabela salva em: {table_path}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle('Comparação de Desempenho dos Modelos (Média dos Folds)', fontsize=16)

    report_df['RMSE (kW)'].plot(kind='bar', ax=axes[0], color='skyblue', title='RMSE (Quanto Menor, Melhor)')
    axes[0].set_ylabel('RMSE (kW)'); axes[0].tick_params(axis='x', rotation=45)

    report_df['MAE (kW)'].plot(kind='bar', ax=axes[1], color='salmon', title='MAE (Quanto Menor, Melhor)')
    axes[1].set_ylabel('MAE (kW)'); axes[1].tick_params(axis='x', rotation=45)

    report_df['R² Score'].plot(kind='bar', ax=axes[2], color='lightgreen', title='R² Score (Quanto Maior, Melhor)')
    axes[2].set_ylabel('R² Score'); axes[2].tick_params(axis='x', rotation=45)
    min_r2 = report_df['R² Score'].min()
    axes[2].set_ylim(bottom=max(0, min_r2 - 0.1), top=1.0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, 'comparison_report.png')
    plt.savefig(plot_path)
    print(f"Gráfico comparativo salvo em: {plot_path}")
    plt.close()

if __name__ == "__main__":
    generate_comparison_report()