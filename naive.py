import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Load dataset
banknotes = pd.read_csv(
    'data/data.csv',
    names=[
        'x0',
        'x1',
        'x2',
        'x3',
        'y0',
    ]
)

# Split train/test sets
train = banknotes.sample(frac=0.8)
test = banknotes.drop(train.index)

# Split training features/labels
train_features = train.copy()
train_labels = train_features.pop('y0')

# Split test features/labels
test_features = test.copy()
test_labels = test_features.pop('y0')

# Define a very simple 
banknotes_model = tf.keras.Sequential([
  layers.Dense(16),
  layers.Dense(1, activation='sigmoid'),
])

banknotes_model.compile(
    loss = tf.losses.MeanSquaredError(),
    optimizer = tf.optimizers.Adam(),
    metrics=['accuracy'],
)

# Train model with training set
banknotes_model.fit(train_features, train_labels, epochs=10)

# Evaluate model on test set
test_loss, test_acc = banknotes_model.evaluate(test_features, test_labels)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")