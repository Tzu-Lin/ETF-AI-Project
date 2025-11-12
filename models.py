# models.py (請將此版本完整覆蓋您的檔案)

import joblib
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input # 新增 Input

# --- 基礎模版 ---
class BaseModel(ABC):
    """所有模型的抽象基礎類別，定義了統一的接口。"""
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        """訓練模型"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        """進行預測，並返回預測結果與機率"""
        raise NotImplementedError

# --- 傳統機器學習模型 ---
class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        # RandomForest's predict_proba returns shape (n_samples, n_classes)
        # e.g., [[prob_down, prob_up], [prob_down, prob_up], ...]
        probabilities = self.model.predict_proba(X_test)
        
        # 我們只取 "上漲" 的機率 (索引為 1 的那一欄)
        prob_up = probabilities[:, 1]
        prob_down = probabilities[:, 0]

        # 為了格式統一，我們把它變成 (n_samples, 1) 的形狀
        # 這樣就和 LSTM 的輸出格式一樣了
        prob_up = prob_up.reshape(-1, 1)
        prob_down = prob_down.reshape(-1, 1)

        return predictions, prob_up, prob_down

# --- 深度學習模型的基礎類別 ---
class DeepLearningModel(BaseModel):
    """深度學習模型的通用類別，處理訓練和預測邏輯。"""
    def __init__(self):
        super().__init__()
    
    def train(self, X_train, y_train, epochs=20, batch_size=32):
        print("開始訓練深度學習模型...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        print("訓練完成。")

    def predict(self, X_test):
        probabilities_up = self.model.predict(X_test).flatten() # 將輸出的(n, 1)陣列壓平成(n,)
        probabilities_down = 1 - probabilities_up
        predictions = (probabilities_up > 0.5).astype(int)
        return predictions, probabilities_up, probabilities_down

# --- 教授要求的四種 LSTM 變體 ---
class SingleLayerLSTM(DeepLearningModel):
    def __init__(self, input_shape):
        self.model = Sequential([
            Input(shape=input_shape), # 獨立的 Input 層
            LSTM(50),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

class DoubleLayerLSTM(DeepLearningModel):
    def __init__(self, input_shape):
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True), 
            LSTM(50),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        
class SingleLayerBiLSTM(DeepLearningModel):
    def __init__(self, input_shape):
        self.model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(50)),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

class DoubleLayerBiLSTM(DeepLearningModel):
    def __init__(self, input_shape):
        self.model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(50, return_sequences=True)),
            Bidirectional(LSTM(50)),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')