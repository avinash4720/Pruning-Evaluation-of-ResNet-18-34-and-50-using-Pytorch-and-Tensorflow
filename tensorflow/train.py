

import tensorflow as tf

# import wandb

# wandb.init(project="test-project", entity="resnetp",name="resnet18")

# wandb.init(project="test-project", entity="resnetp")
# wandb.config = {
#   "learning_rate": 0.0001,
#   "epochs": 10,
#   "batch_size": 128
# }

lr = 0.0001
batch_size = 128
EPOCHS = 10
def train(model):
  # Build your model here
  model_name = f"""tensor_{model.__name__}"""
  model = model()
  optimizer = tf.keras.optimizers.Adam(lr)

  # Load and prepare the cifar10 dataset.
  cifar10 = tf.keras.datasets.cifar10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  y_train, y_test = tf.reshape(y_train, (-1,)), tf.reshape(y_test, (-1,))

  # Use tf.data to batch and shuffle the dataset
  train_ds = tf.data.Dataset.from_tensor_slices(
          (x_train, y_train)).shuffle(100).batch(batch_size)
  test_ds = tf.data.Dataset.from_tensor_slices(
          (x_test, y_test)).batch(batch_size)

  # Choose an optimizer and loss function for training
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

  # Select metrics to measure the loss and the accuracy of the model
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  # Use tf.GradientTape to train the model.
  @tf.function
  def train_step(images, labels):
      with tf.GradientTape() as tape:
          predictions = model(images, training=True)
          # print("=> label shape: ", labels.shape, "pred shape", predictions.shape)
          loss = loss_object(labels, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      train_loss(loss)
      train_accuracy(labels, predictions)

  @tf.function
  def test_step(images, labels):
      predictions = model(images)
      t_loss = loss_object(labels, predictions)
      test_loss(t_loss)
      test_accuracy(labels, predictions)

  for epoch in range(EPOCHS):
      for images, labels in train_ds:
          train_step(images, labels)

      for test_images, test_labels in test_ds:
          test_step(test_images, test_labels)

      template = '=> Epoch {}, Loss: {:.4}, Accuracy: {:.2%}, Test Loss: {:.4}, Test Accuracy: {:.2%}'
      print(template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result(),
                            test_loss.result(),
                            test_accuracy.result()))
      # wandb.log({ "Epoch": epoch, "Train Loss": train_loss.result(), "Train Acc": train_accuracy.result(), "Test Loss": test_loss.result(), "Test Accuracy": test_accuracy.result()})
      # Reset the metrics for the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()
      
model.save_weights(f"""saved_models/{model_name}""")