import tensorflow.keras as keras

# dataset
# --------------------------------------

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.
test_images = test_images / 255.

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# model
# --------------------------------------

model = keras.models.load_model('saved_model/model')

model.summary()

loss, accuracy = model.evaluate(
    test_images,
    test_labels,
    verbose = 2
)

print('loss: ', loss)
print('accuracy: ', accuracy)
