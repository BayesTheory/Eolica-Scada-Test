# Modelos/1Arima.py
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    def __init__(self, order, **kwargs): # **kwargs para ignorar params extras
        self.order = order
        self.model_fit = None

    def fit(self, train_data):
        self.model_fit = ARIMA(train_data, order=self.order).fit()

    def predict(self, steps):
        return self.model_fit.forecast(steps=steps)