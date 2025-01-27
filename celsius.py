import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_data():
    celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
    return celsius, fahrenheit

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(units=1)
    ])
    model.compile(optimizer=tf.optimizers.Adam(0.1), loss='mean_squared_error')
    return model

def train_model(model, celsius, fahrenheit):
    history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)
    return history

def predict(model, value):
    prediction = model.predict(np.array([value]))
    return prediction

def main(x):
    celsius, fahrenheit = load_data()
    model = build_model()
    train_model(model, celsius, fahrenheit)
    predicted = predict(model, x)
    print(predicted)

if __name__ == "__main__":
    main(2)