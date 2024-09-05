from sklearn.datasets import load_wine
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
data = load_wine()
X = data.data
y = data.target



tf.random.set_seed(1234)
model = Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')        
        ### END CODE HERE ### 
    ], name = "my_model" 
)

model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X,y,
    epochs=100 #changing this from 100 to 200 to 1000 made the error count decrease
)

count = 0
for i in range(len(X)):
    prediction = model.predict(X[[i]])
    if (np.argmax(prediction) != y[i]):
        count += 1
        print(f"Prediction:\n{prediction}")
        print(f"Largest Prediction index: {np.argmax(prediction)}")
        print(f"Actual: {y[i]}")

print(f"Error Total: {count}")

# What I still do not understand is how many layers to use, how many units to use, and how many epochs to use