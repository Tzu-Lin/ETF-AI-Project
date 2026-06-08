# models.py  (完整覆蓋)
# 分類模型(原有, 不動) + 迴歸模型(新增) 並存
import joblib
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input


# ============ 基礎模版 ============
class BaseModel(ABC):
    def __init__(self):
        self.model = None
    @abstractmethod
    def train(self, X_train, y_train): raise NotImplementedError
    @abstractmethod
    def predict(self, X_test): raise NotImplementedError


# ============ 傳統機器學習(分類) ============
class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        prob_up = probabilities[:, 1].reshape(-1, 1)
        prob_down = probabilities[:, 0].reshape(-1, 1)
        return predictions, prob_up, prob_down


# ============ 深度學習(分類)基礎 ============
class DeepLearningModel(BaseModel):
    def __init__(self): super().__init__()
    def train(self, X_train, y_train, epochs=20, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    def predict(self, X_test):
        probabilities_up = self.model.predict(X_test).flatten()
        probabilities_down = 1 - probabilities_up
        predictions = (probabilities_up > 0.5).astype(int)
        return predictions, probabilities_up, probabilities_down


# ============ 四種 LSTM 變體(分類, 原有) ============
class SingleLayerLSTM(DeepLearningModel):
    def __init__(self, input_shape):
        self.model = Sequential([Input(shape=input_shape), LSTM(50),
                                 Dense(1, activation='sigmoid')])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

class DoubleLayerLSTM(DeepLearningModel):
    def __init__(self, input_shape):
        self.model = Sequential([Input(shape=input_shape),
                                 LSTM(50, return_sequences=True), LSTM(50),
                                 Dense(1, activation='sigmoid')])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

class SingleLayerBiLSTM(DeepLearningModel):
    def __init__(self, input_shape):
        self.model = Sequential([Input(shape=input_shape), Bidirectional(LSTM(50)),
                                 Dense(1, activation='sigmoid')])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

class DoubleLayerBiLSTM(DeepLearningModel):
    def __init__(self, input_shape):
        self.model = Sequential([Input(shape=input_shape),
                                 Bidirectional(LSTM(50, return_sequences=True)),
                                 Bidirectional(LSTM(50)),
                                 Dense(1, activation='sigmoid')])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')


# ============ 深度學習(迴歸)基礎 ============ 【新增】
class DeepLearningRegressor(BaseModel):
    """迴歸版: 輸出連續值, 損失用 mse, 無 sigmoid。"""
    def __init__(self): super().__init__()
    def train(self, X_train, y_train, epochs=20, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    def predict(self, X_test):
        return self.model.predict(X_test, verbose=0).flatten()


# ============ 四種 LSTM 變體(迴歸, 新增) ============ 【新增】
class SingleLayerLSTMRegressor(DeepLearningRegressor):
    def __init__(self, input_shape):
        self.model = Sequential([Input(shape=input_shape), LSTM(50), Dense(1)])
        self.model.compile(optimizer='adam', loss='mse')

class DoubleLayerLSTMRegressor(DeepLearningRegressor):
    def __init__(self, input_shape):
        self.model = Sequential([Input(shape=input_shape),
                                 LSTM(50, return_sequences=True), LSTM(50), Dense(1)])
        self.model.compile(optimizer='adam', loss='mse')

class SingleLayerBiLSTMRegressor(DeepLearningRegressor):
    def __init__(self, input_shape):
        self.model = Sequential([Input(shape=input_shape), Bidirectional(LSTM(50)), Dense(1)])
        self.model.compile(optimizer='adam', loss='mse')

class DoubleLayerBiLSTMRegressor(DeepLearningRegressor):
    def __init__(self, input_shape):
        self.model = Sequential([Input(shape=input_shape),
                                 Bidirectional(LSTM(50, return_sequences=True)),
                                 Bidirectional(LSTM(50)), Dense(1)])
        self.model.compile(optimizer='adam', loss='mse')