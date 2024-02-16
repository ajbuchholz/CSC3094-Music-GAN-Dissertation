import pandas as pd
import tensorflow as tf
import numpy as np

def normalise_pitch(notes, vocab_size):
    """
    Normalizes the pitch of the notes by the vocabulary size.

    Parameters:
        notes (tf.SymbolicTensor): An array of notes, where each note is represented as a vector of [pitch, step, duration].
        vocab_size (int): The size of the pitch vocabulary.

    Returns:
        np.ndarray: The normalized notes array.
    """
    return notes / [vocab_size, 1.0, 1.0]

def split_input_label(sequence, vocabulary_size):
    """
    Splits a sequence of notes into input features and labels for model training.

    Parameters:
        sequence (np.ndarray): A sequence of notes to be split.
        vocabulary_size (int): The size of the pitch vocabulary for normalization.

    Returns:
        tuple: A tuple containing the input sequence (with normalized pitch) and a dictionary of labels
               for 'pitch', 'step', and 'duration'.
    """
    input_sequence = sequence[:-1]
    label_sequence = sequence[-1]
    input_sequence = normalise_pitch(input_sequence, vocabulary_size)
    labels = {'pitch': label_sequence[0], 'step': label_sequence[1], 'duration': label_sequence[2]}
    return input_sequence, labels

def generate_note_sequences(all_normalized_notes, sequence_length, vocabulary_size=128):
    """
    Generates training sequences from normalized notes data.

    Parameters:
        all_normalized_notes (dict): A dictionary with keys 'pitch', 'step', 'duration', each mapping to a list of normalized note values.
        sequence_length (int): The length of the sequences to generate.
        vocabulary_size (int, optional): The size of the pitch vocabulary for normalization. Defaults to 128.

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing tuples of input sequences and labels.
    """

    selected_attributes = ['pitch', 'step', 'duration']
    training_data = np.stack([all_normalized_notes[attribute] for attribute in selected_attributes], axis=1)
    dataset = tf.data.Dataset.from_tensor_slices(training_data)
    sequence_length += 1  # Adjust for target sequence length
    sequences = dataset.window(size=sequence_length, shift=1, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(sequence_length, drop_remainder=True))
    return sequences.map(lambda sequence: split_input_label(sequence, vocabulary_size), num_parallel_calls=tf.data.AUTOTUNE)

def custom_mse_positive(y_true, y_pred):
    """
    Custom Mean Squared Error loss function that penalizes positive errors more.

    Parameters:
        y_true (tf.Tensor): The true values.
        y_pred (tf.Tensor): The predicted values by the model.

    Returns:
        tf.Tensor: The computed loss value.
    """
    mse = tf.reduce_mean((y_true - y_pred) ** 2)
    positive_pressure = 10 * tf.reduce_mean(tf.maximum(-y_pred, 0.0))
    return mse + positive_pressure

def define_rnn_model(sequence_length=25, loss_weight_pitch=0.1, loss_weight_step=1.0, loss_weight_duration=1.0):
    """
    Defines and compiles a recurrent neural network (RNN) model for music generation.

    This function creates an RNN model with a single LSTM layer followed by three Dense layers
    for predicting pitch, step (time until the next note), and duration of music notes.

    Parameters:
        sequence_length (int, optional): The length of the input sequences. Defaults to 25.
        loss_weight_pitch (float, optional): The loss weight for the pitch prediction. Defaults to 0.05.
        loss_weight_step (float, optional): The loss weight for the step prediction. Defaults to 1.0.
        loss_weight_duration (float, optional): The loss weight for the duration prediction. Defaults to 1.0.

    Returns:
        tf.keras.Model: The compiled RNN model ready for training.
    """

    input_shape = (sequence_length, 3)
    inputs = tf.keras.Input(shape=input_shape)
    # First LSTM layer
    x = tf.keras.layers.LSTM(256)(inputs)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        "pitch": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "step": custom_mse_positive,
        "duration": custom_mse_positive, 
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(
        loss=loss,
        loss_weights={'pitch': loss_weight_pitch, 'step': loss_weight_step, 'duration': loss_weight_duration},
        optimizer=optimizer,
    )
    model.summary()

    return model

def train_rnn_model(model, training_song_dataset, number_of_epoch):
    """
    Trains a recurrent neural network (RNN) model on a dataset of songs.

    This function takes a pre-defined RNN model and a dataset of songs for training. It
    trains the model for a specified number of epochs, saving the model's weights at the
    end of each epoch.

    Parameters:
        model (tf.keras.Model): The RNN model to be trained. 
        training_song_dataset (tf.data.Dataset): A TensorFlow dataset object containing the training data.
        number_of_epoch (int): The number of epochs to train the model. 
    Returns:
        tf.keras.callbacks.History: A History object.
    """
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='model_weights/checkpoint_{epoch}.hdf5',
        save_weights_only=True,
        monitor='loss', 
        mode='min',
        save_best_only=True
    )
    history = model.fit(training_song_dataset, epochs=number_of_epoch, callbacks=callback)
    return history

def predict_next_note(notes, model, temperature=1.0, subtract_step=0.0, subtract_duration=0.0):
    """
    Predicts the next note in a sequence of notes using a given model.

    This function takes a sequence of notes, a trained model, and a temperature parameter
    to control the randomness of the prediction. It processes the input notes to predict
    the pitch, step (time until the next note), and duration of the next note in the sequence.

    Parameters:
        notes (tf.Tensor): A tensor containing the input sequence of notes.
        model (tf.keras.Model): The trained model used for prediction.
        temperature (float, optional): A factor to control the randomness of the prediction
            by scaling the pitch logits. Default is 1.0, which represents no scaling.
        subtract_step (float, optional): A value to control the step amount when it's too high.
        subtract_duration (float, optional): A value to control the duration amount when it's too high.

    Returns:
        tuple: A tuple containing three elements:
            - int: The predicted pitch of the next note.
            - float: The predicted step (time until the next note) as a float.
            - float: The predicted duration of the next note as a float.
    """
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs, verbose=1)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature # Add variation
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration-subtract_duration, axis=-1)
    step = tf.squeeze(step-subtract_step, axis=-1)
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)
