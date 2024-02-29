import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import glob
import shutil
import sys
import pandas as pd
import numpy as np
from keras import layers, models, losses, optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import generate_note_sequences_pitch, normalise_song_pitch, array_to_midi_pretty, visualise_song, plot_distributions

NUMBER_OF_EPOCHS = 100
SEQUENCE_LENGTH = 100
TRAIN_MODEL = False
NUMBER_FILES_TO_PARSE = 10
NUMBER_OF_PREDICTIONS = 100
VOCABULARY_SIZE = 128
BATCH_SIZE = 64

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
    positive_pressure = 15 * tf.reduce_mean(tf.maximum(-y_pred, 0.0))
    return mse + positive_pressure

def define_rnn_model(sequence_length=25, loss_weight_pitch=0.1, loss_weight_step=2.0, loss_weight_duration=2.0):
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
    inputs = layers.Input(shape=input_shape)

    # LSTM layer
    x = layers.LSTM(256)(inputs)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(16)(x)
    x = layers.BatchNormalization()(x)

    # Output layers
    pitch_output = layers.Dense(128, name='pitch')(x)
    step_output = layers.Dense(1, name='step')(x)
    duration_output = layers.Dense(1, name='duration')(x)

    # Model definition
    model = models.Model(inputs=inputs, outputs={"pitch": pitch_output, "step": step_output, "duration": duration_output})

    loss = {
        "pitch": losses.SparseCategoricalCrossentropy(from_logits=True),
        "step": custom_mse_positive,
        "duration": custom_mse_positive, 
    }

    optimizer = optimizers.Adam(learning_rate=0.005)
    model.compile(
        loss=loss,
        loss_weights={'pitch': loss_weight_pitch, 'step': loss_weight_step, 'duration': loss_weight_duration},
        optimizer=optimizer,
    )

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
        filepath='rnnpitch_checkpoints/checkpoint_{epoch}.weights.h5',
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

if __name__ == '__main__':
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Load our MIDI files from the directory
    midi_files = glob.glob("../data/classical-piano/*.mid")

    # Normalise our dataset
    normalised_notes = []
    for file in midi_files[:NUMBER_FILES_TO_PARSE]:
        notes = normalise_song_pitch(file)
        normalised_notes.append(notes)
    all_normalised_notes = pd.concat(normalised_notes)

    # Set the sequence length for each example
    key_order = ['pitch', 'step', 'duration']
    sequence_song_dataset = generate_note_sequences_pitch(all_normalised_notes, SEQUENCE_LENGTH, VOCABULARY_SIZE)

    # Define RNN Model
    buffer_size = len(all_normalised_notes) - SEQUENCE_LENGTH  # the number of items in the dataset
    training_song_dataset = sequence_song_dataset.shuffle(buffer_size).batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
    model = define_rnn_model(sequence_length=SEQUENCE_LENGTH)
    
    if TRAIN_MODEL:
        # Train Model
        shutil.rmtree('rnnpitch_checkpoints')
        history = train_rnn_model(model, training_song_dataset, number_of_epoch=NUMBER_OF_EPOCHS)
        plt.plot(history.epoch, history.history['loss'], label='total loss')
        plt.show()
    else:
        # Load Model
        try:
            model.load_weights(f'final_weights/weights_pitch.weights.h5')
        except:
            sys.exit("Weight File Not Found")

    # Generate Song
    input_notes = np.stack([all_normalised_notes[key] for key in key_order], axis=1)[:SEQUENCE_LENGTH] / np.array([VOCABULARY_SIZE, 1, 1])
    generated_notes = []
    start = 0
    for _ in range(NUMBER_OF_PREDICTIONS):
        pitch, step, duration = predict_next_note(input_notes, model, temperature=6.5, subtract_step=2.1, subtract_duration=2.9)
        start += step
        end = start + duration
        generated_note = (pitch, step, duration, start, end)
        generated_notes.append(generated_note)
        input_notes = np.vstack([input_notes[1:], np.array(generated_note[:3])])

    generated_notes = pd.DataFrame(generated_notes, columns=[*key_order, 'start', 'end'])

    # Save Generate Song
    visualise_song(generated_notes)
    plot_distributions(generated_notes)
    array_to_midi_pretty(generated_notes, "output-rnn-pitch.mid")
