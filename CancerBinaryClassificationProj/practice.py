from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data = load_breast_cancer()
X = data.data
print(X)
y = data.target

model = Sequential(
    [               
        tf.keras.layers.Dense(units=25, activation="sigmoid"),
        tf.keras.layers.Dense(units=15, activation="sigmoid"),
        tf.keras.layers.Dense(units=1, activation="sigmoid"), 
    ], name = "my_model" 
)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X,y,
    epochs=75
)


prediction1 = model.predict(X[[240]])
print(f"Actual Value: {y[240]}")
print(f"Predicted Value: {prediction1}")
print(" ")
prediction2 = model.predict(X[[100]])
print(f"Actual Value: {y[100]}")
print(f"Predicted Value: {prediction2}")

if (prediction1 < 0.5):
    prediction1 = 0 
else:
    prediction1 = 1

if (prediction2 < 0.5):
    prediction2 = 0 
else:
    prediction2 = 1

print(f"Cancer 1 is {data.target_names[prediction1]}")
print(f"Cancer 2 is {data.target_names[prediction2]}")

#Questions that have arisen from this work
#1: Systematic way to determine number of epochs (what are they and how do they work?)
#2: how does model.compile work
#3: Systematic way to determine the number of layers I should use