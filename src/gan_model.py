import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import shutil
import glob
import tensorflow as tf
import numpy as np
from keras import layers, optimizers, models
from utils import normalise_song, generate_note_sequences, create_midi_chords

NUMBER_FILES_TO_PARSE = 10
EPOCHS = 500
BATCH_SIZE = 128
SEQUENCE_LENGTH = 100
SEQUENCE_SHAPE = (SEQUENCE_LENGTH, 1)
LATENT_DIMENSION = 1000
NUMBER_OF_NOTES = 100

def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=LATENT_DIMENSION))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.15))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(np.prod(SEQUENCE_SHAPE), activation='tanh'))
    model.add(layers.Reshape(SEQUENCE_SHAPE))
    noise = layers.Input(shape=(LATENT_DIMENSION,))
    seq = model(noise)

    return models.Model(noise, seq)

def build_discriminator():
    model = models.Sequential()
    model.add(layers.LSTM(512, input_shape=SEQUENCE_SHAPE, return_sequences=True))
    model.add(layers.Bidirectional(layers.LSTM(512)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    sequence = layers.Input(shape=SEQUENCE_SHAPE)
    validity = model(sequence)

    return models.Model(sequence, validity)

def build_gan(generator, discriminator):
    discriminator.trainable = False
    noise = layers.Input(shape=(LATENT_DIMENSION,))
    generated_sequence = generator(noise)
    validity = discriminator(generated_sequence)
    gan_model = models.Model(noise, validity)
    gan_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002, 0.5))
    return gan_model

def train(gan, generator, discriminator, songs_tensor, vocabulary_size, assign_int_to_note):
    if os.path.exists("gan_checkpoints"):
        shutil.rmtree("gan_checkpoints")
        os.makedirs("gan_checkpoints")
    else:
        os.makedirs("gan_checkpoints")

    for epoch in range(EPOCHS):
        real_seqs = songs_tensor[np.random.randint(0, songs_tensor.shape[0], BATCH_SIZE)]
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIMENSION))
        fake_seqs = generator.predict(noise, verbose=0)
        d_loss_real = discriminator.train_on_batch(real_seqs, np.ones((BATCH_SIZE, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_seqs, np.zeros((BATCH_SIZE, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIMENSION))
        g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

        print (f"{epoch} Discriminator loss: {d_loss[0]:.5f}, acc.: {100*d_loss[1]:.2f}% | Generator loss: {g_loss:.5f}")

        if epoch % 10 == 0:
            predictions = generator.predict(noise)
            predictions = [x * (vocabulary_size / 2) + (vocabulary_size / 2) for x in predictions[0]]
            predictions = [assign_int_to_note[int(x[0])] for x in predictions if int(x[0]) in assign_int_to_note]
            create_midi_chords(predictions, f'gan_checkpoints/generated_song_{epoch}')

if __name__ == '__main__':
    midi_files = glob.glob("../data/classical-piano/*.mid")
        
    notes = []
    for file in midi_files[:NUMBER_FILES_TO_PARSE]:
        normalised_notes = normalise_song(file)
        notes.extend(normalised_notes)
    
    vocabulary_size = len(set(notes))
    network_input, _, assign_int_to_note = generate_note_sequences(notes, SEQUENCE_LENGTH, vocabulary_size)
    
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

    train(gan, generator, discriminator, network_input, vocabulary_size, assign_int_to_note)
    noise = np.random.normal(0, 1, (1, LATENT_DIMENSION))
    predictions = generator.predict(noise)
    predictions = [x * (vocabulary_size / 2) + (vocabulary_size / 2) for x in predictions[0]]
    predictions = [assign_int_to_note[int(x[0])] for x in predictions if int(x[0]) in assign_int_to_note]
    create_midi_chords(predictions, 'generated_song')
