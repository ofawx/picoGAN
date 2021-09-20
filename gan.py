import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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

# Make real banknotes train
banknotes_real = banknotes[banknotes['y0'] == 0]
train = banknotes_real.sample(frac=0.8)
train_features = train.copy()
train_labels = train_features.pop('y0')
train_dataset = tf.data.Dataset.from_tensor_slices(train_features).shuffle(10000).batch(64) # make a tf dataset for ease

# Use remaining real notes as test
test_real = banknotes_real.drop(train.index)

# Make fake banknote test ONLY
banknotes_fake = banknotes[banknotes['y0'] == 1]
test_fake = banknotes_fake.sample(frac=0.2)

# Combine fake test sets
test = pd.concat([test_real,test_fake])
test = test.sample(frac=1) # shuffle randomly
test_features = test.copy()
test_labels = test_features.pop('y0')
test_feat_tensor = tf.convert_to_tensor(test_features) # make a tf tensor for easy use later

# Make Generator
gen = tf.keras.Sequential()
gen.add(layers.Dense(512, use_bias=False, input_shape=(512,)))
gen.add(layers.Dense(128, activation='sigmoid'))
gen.add(layers.Dense(32))
gen.add(layers.Dense(4))
assert gen.output_shape == (None, 4) # Note: None is the batch size

# Make discriminator
discr = tf.keras.Sequential()
discr.add(layers.Dense(4, input_shape=(4,)))
discr.add(layers.Dense(64))
discr.add(layers.Dense(128))
discr.add(layers.Dense(64))
discr.add(layers.Dense(32))
discr.add(layers.Dense(1, activation='sigmoid'))
assert discr.output_shape == (None, 1) # Note: None is the batch size

# Define loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.zeros_like(real_output), real_output)
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.zeros_like(fake_output), fake_output)

def binary_accuracy(pred, labels):
    return accuracy_score(labels, tf.round(pred))

# Define optimizers, one per model
generator_optimizer = tf.keras.optimizers.Adam(1e-6)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-6)

# Define Training Job
@tf.function
def train_step(data):
    
    # Create some random noise
    noise = tf.random.normal([64, 512]) # Batch x Input

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Feed the random noise through the generator (to get fake banknotes)
        generated_data = gen(noise, training=True)
        # Feed the real banknotes through the discriminator
        real_output = discr(data, training=True)
        # Feed the fake banknotes through the discriminator
        fake_output = discr(generated_data, training=True)

        # How well did each model perform?
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Back propagate error into models
    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discr.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discr.trainable_variables))

    return gen_loss, disc_loss

# # Save model ckpts
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(
#     generator_optimizer=generator_optimizer,
#     discriminator_optimizer=discriminator_optimizer,
#     generator=gen,
#     discriminator=discr
# )

# Run the train
for epoch in range(600):
    # Do the actual training
    for image_batch in train_dataset:
        gen_loss, disc_loss = train_step(image_batch)

    # Every 10 epochs...
    if (epoch + 1) % 10 == 0:
        # checkpoint.save(file_prefix = checkpoint_prefix)

        # Evaluate test accuracy and print
        test_acc = binary_accuracy(
            discr(test_feat_tensor),
            test_labels,
        )
        print(f'Epoch {epoch + 1}, test accuracy: {test_acc}, gen_loss: {gen_loss}, disc_loss: {disc_loss}')


# Overall performance
print(f'Final test accuracy: {binary_accuracy(discr(test_feat_tensor), test_labels)}')