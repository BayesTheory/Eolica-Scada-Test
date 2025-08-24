# Modelos/2XGboosting.py
import xgboost as xgb

class XGBoostModel:
    def __init__(self, **params):
        # Ignora parâmetros de PyTorch que não são usados aqui
        pytorch_params = ['hidden_size', 'n_layers', 'dropout', 'epochs', 'batch_size', 'learning_rate']
        xgb_params = {k: v for k, v in params.items() if k not in pytorch_params}
        self.model = xgb.XGBRegressor(**xgb_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)