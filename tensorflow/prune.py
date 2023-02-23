import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tempfile
import tensorflow_model_optimization as tfmot
import numpy as np




def prune(model):  
    # Load and prepare the cifar10 dataset.
    # Model configuration
    img_width, img_height = 32, 32
    batch_size = 128
    no_classes = 10
    validation_split = 0.2
    verbosity = 1
    pruning_epochs = 30



    input_shape = (img_width, img_height, 1)

    cifar10 = tf.keras.datasets.cifar10
    (input_train, target_train), (input_test, target_test) = cifar10.load_data()

    # Reshape data for ConvNet
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
    input_shape = (img_width, img_height, 1)

    # Choose an optimizer and loss function for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Select metrics to measure the loss and the accuracy of the model
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    # Parse numbers as floats
    input_train = input_train.astype('float32')
    input_test = input_test.astype('float32')

    # Normalize [0, 255] into [0, 1]
    input_train = input_train / 255
    input_test = input_test / 255

    # Convert target vectors to categorical targets
    target_train = tensorflow.keras.utils.to_categorical(target_train, no_classes)
    target_test = tensorflow.keras.utils.to_categorical(target_test, no_classes)
    
    model = model()

    # Compile the model
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                optimizer=tensorflow.keras.optimizers.Adam(),
                metrics=['accuracy'])

    # Load functionality for adding pruning wrappers
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Finish pruning after 10 epochs
    num_images = input_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs

    # Define pruning configuration
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.875,
                                                                begin_step=0.2*end_step,
                                                                end_step=end_step)
    }
    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # Recompile the model
    model_for_pruning.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                optimizer=tensorflow.keras.optimizers.Adam(),
                metrics=['accuracy'])

    # Model callbacks
    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
    ]

    # Fitting data
    model_for_pruning.fit(input_train, target_train,
                        batch_size=batch_size,
                        epochs=pruning_epochs,
                        verbose=verbosity,
                        callbacks=callbacks,
                        validation_split=validation_split)

    # Generate generalization metrics
    score_pruned = model_for_pruning.evaluate(input_test, target_test, verbose=0)
    print(f'Pruned CNN - Test loss: {score_pruned[0]} / Test accuracy: {score_pruned[1]}')

    # Export the model
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    _, pruned_keras_file = tempfile.mkstemp('.h5')
    save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print(f'Pruned model saved: {pruned_keras_file}')

    def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)

    print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))