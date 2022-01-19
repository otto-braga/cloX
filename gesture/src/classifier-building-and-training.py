import os
import tensorflow.keras as keras

train = True
batch_size = 256
epochs = 40

weights_path = "training/weights-epoch_{epoch:04d}.ckpt"
log_path = 'training/log.csv'

# network
# --------------------------------------

def lenet():
    return keras.Sequential(
        [
            keras.layers.Conv2D(
                filters = 6,
                kernel_size = 5,
                strides = (1,1),
                padding = 'same',
                activation = 'sigmoid'
            ),
            keras.layers.AvgPool2D(
                pool_size = 2,
                strides = 2
            ),
            keras.layers.Conv2D(
                filters = 16,
                kernel_size = 5,
                activation = 'sigmoid'
            ),
            keras.layers.AvgPool2D(
                pool_size = 2,
                strides = 2
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(
                units = 120,
                activation = 'sigmoid'
            ),
            keras.layers.Dense(
                units = 84,
                activation = 'sigmoid'
            ),
            keras.layers.Dense(
                units = 10
            )
        ]
    )

model = lenet()

model.compile(
    optimizer = 'adam',
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

model.build((1, 28, 28, 1))

model.summary()

# dataset
# --------------------------------------

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.
test_images = test_images / 255.

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# training
# --------------------------------------

weights_save_callback = keras.callbacks.ModelCheckpoint(
    filepath = weights_path,
    save_weights_only = True,
    verbose = 1
)

log_callback = keras.callbacks.CSVLogger(
    filename = log_path,
    separator = ',',
    append = False
)

if train:
    log = model.fit(
        x = train_images,
        y = train_labels,
        batch_size = batch_size,
        epochs = epochs,
        callbacks = [weights_save_callback, log_callback]
    )

    print(log.history.keys())
    print(log.history['loss'])
    print(log.history['accuracy'])

    model.save('training/saved_model/model')
