import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob
import shutil
import numpy as np
from keras import layers, models
from utils import normalise_song, generate_note_sequences, create_midi_chords

NUMBER_FILES_TO_PARSE = 50
LATENT_DIMENSION = 1000
NUMBER_OF_EPOCHS = 500
BATCH_SIZE = 16
SEQUENCE_LENGTH = 100
SEQUENCE_SHAPE = (SEQUENCE_LENGTH, 1)

def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(1024, input_dim=LATENT_DIMENSION))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod(SEQUENCE_SHAPE), activation='tanh'))
    model.add(layers.Reshape(SEQUENCE_SHAPE))
    noise = layers.Input(shape=(LATENT_DIMENSION,))
    seq = model(noise)

    return models.Model(noise, seq)

def build_discriminator():
    model = models.Sequential()
    model.add(layers.LSTM(512, input_shape=SEQUENCE_SHAPE, return_sequences=True))
    model.add(layers.Bidirectional(layers.LSTM(512)))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.5))
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
    gan_model.compile(loss='binary_crossentropy', optimizer="adam")
    return gan_model

def train(epochs, batch_size, generator, discriminator, gan, network_input, vocabulary_size, notes):
    if os.path.exists("gan_checkpoints"):
        shutil.rmtree("gan_checkpoints")
        os.makedirs("gan_checkpoints")
    else:
        os.makedirs("gan_checkpoints")

    for epoch in range(epochs):
        # Get a random batch of real sequences
        real_seqs = network_input[np.random.randint(0, network_input.shape[0], batch_size)]

        # Generate a batch of fake sequences
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIMENSION))
        fake_seqs = generator.predict(noise, verbose=0)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_seqs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_seqs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator 
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIMENSION))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print the progress
        print (f"{epoch} Discriminator loss: {d_loss[0]:.5f}, acc.: {100*d_loss[1]:.2f}% | Generator loss: {g_loss:.5f}")

        # Save a checkpoint every 5 epochs
        if epoch % 5 == 0:
            epoch_dir = os.path.join("gan_checkpoints", f"epoch_{epoch}")
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)

            # Save generator and discriminator weights
            generator_weights_path = os.path.join(epoch_dir, "generator_weights.weights.h5")
            discriminator_weights_path = os.path.join(epoch_dir, "discriminator_weights.weights.h5")
            generator.save_weights(generator_weights_path)
            discriminator.save_weights(discriminator_weights_path)

            # Save output MIDI file
            noise = np.random.normal(0, 1, (1, LATENT_DIMENSION))
            generator_model = build_generator()  # Create a new instance of the generator model
            generator_model.load_weights(f"gan_checkpoints/epoch_{epoch}/generator_weights.weights.h5")
            predictions = generator_model.predict(noise)
            prediction_notes = [x * (vocabulary_size / 2) + (vocabulary_size / 2) for x in predictions[0]]
            pitchnames = sorted(set(item for item in notes))
            assign_int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
            prediction_notes_mapped = [assign_int_to_note[int(x[0])] for x in prediction_notes if int(x[0]) in assign_int_to_note]
            create_midi_chords(prediction_notes_mapped, f"gan_checkpoints/epoch_{epoch}/output.mid")

if __name__ == '__main__':
    midi_files = glob.glob("../data/classical-piano/*.mid")
        
    notes = []
    for file in midi_files[:NUMBER_FILES_TO_PARSE]:
        print("Parsing file: ", file)
        normalised_notes = normalise_song(file)
        notes.extend(normalised_notes)
    
    vocabulary_size = len(set(notes))
    network_input, _, _ = generate_note_sequences(notes, SEQUENCE_LENGTH, vocabulary_size)

    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    gan = build_gan(generator, discriminator)
    train(epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE, generator=generator, discriminator=discriminator, gan=gan, network_input=network_input, vocabulary_size=vocabulary_size, notes=notes)
