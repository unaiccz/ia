import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_data():
    n1 = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2, 6, 10, 20, 40, 80, 160, 320, 640, 1280, 2560], dtype=float)
    n2 = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4, 12, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120], dtype=float)
    return n1, n2

def normalize_data(n1, n2):
    x1 = n1 / 2560.0
    x2 = n2 / 5120.0
    return x1, x2

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(units=10, activation='relu'),
        layers.Dense(units=1)
    ])
    model.compile(optimizer=tf.optimizers.Adam(0.01), loss='mean_squared_error')
    return model

def train_model(model, x1, x2):
    history = model.fit(x1, x2, epochs=1000, verbose=False)
    return history

def predict(model, value):
    value_norm = value / 2560.0
    prediction_norm = model.predict(np.array([value_norm]))
    prediction = prediction_norm * 5120.0
    return prediction

def main(x):
    n1, n2 = load_data()
    x1, x2 = normalize_data(n1, n2)
    model = build_model()
    train_model(model, x1, x2)
    predicted = predict(model, x)
    print(predicted)

if __name__ == "__main__":
    main(14)