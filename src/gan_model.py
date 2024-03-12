import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import shutil
import glob
import tensorflow as tf
import numpy as np
from keras import layers, optimizers, models
from music21 import converter, note, stream, chord, instrument
from utils import normalise_song, generate_note_sequences, create_midi_chords

NUMBER_FILES_TO_PARSE = 200
EPOCHS = 250
BATCH_SIZE = 128
SEQUENCE_LENGTH = 100
SEQUENCE_SHAPE = (SEQUENCE_LENGTH, 1)
LATENT_DIMENSION = 1000
NUMBER_OF_NOTES = 100
NORMALISTION_METHOD = 0 # 0 for Note and Chords, 1 for Frequency, 2 for Note On and Note Off

##############################################################################################

# Normalisation for Frequency
def extract_frequencies(midi_file):
    frequencies = []
    midi = converter.parse(midi_file)
    print(f"Parsing {midi_file}")
    notes_to_parse = midi.flatten().notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            frequencies.append(str(element.pitch.frequency))
        elif isinstance(element, chord.Chord):
            frequencies.append('-'.join(str(pitch.frequency) for pitch in element.pitches))
    
    return frequencies

def frequency_to_midi_number(freq):
    return 69 + 12 * np.log2(float(freq) / 440.0)

def create_midi_from_frequencies(frequencies, filename):
    song_notes = []
    offset = 0
    for freq in frequencies:
        if '-' in freq:  # If it's a chord
            frequencies_in_chord = freq.split('-')
            notes = [note.Note(int(frequency_to_midi_number(current_note)), offset=offset, storedInstrument=instrument.Piano()) for current_note in frequencies_in_chord]
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            song_notes.append(new_chord)
        else:
            new_note = note.Note(int(frequency_to_midi_number(freq)), offset=offset, storedInstrument=instrument.Piano())
            song_notes.append(new_note)
    
    offset += 0.1
    midi_stream = stream.Stream(song_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))

##############################################################################################
    
# Normalisation for Note On and Note Off
def midi_to_events(midi_file):
    midi_stream = converter.parse(midi_file)
    print(f"Parsing {midi_file}")
    events = []
    time_offset = 0

    for part in midi_stream.recurse().getElementsByClass('Stream'):
        for element in part.flatten().notes:
            if isinstance(element, note.Note):
                events.append([1, element.pitch.midi, element.offset + time_offset])
                events.append([0, element.pitch.midi, element.offset + element.duration.quarterLength + time_offset])
            elif isinstance(element, chord.Chord):
                for chord_note in element.pitches:
                    events.append([1, chord_note.midi, element.offset + time_offset])
                    events.append([0, chord_note.midi, element.offset + element.duration.quarterLength + time_offset])

        time_offset += part.duration.quarterLength
    events.sort(key=lambda event: event[2])

    return events

def generate_note_sequences_events(notes_list, sequence_length, vocabulary_size):
    unique_notes = sorted(set(tuple(i) for i in notes_list))
    assign_note_to_int = {note: number for number, note in enumerate(unique_notes)}
    assign_int_to_note = {number: note for note, number in assign_note_to_int.items()}
    network_input = []   
    notes_as_int = [assign_note_to_int[tuple(note)] for note in notes_list]

    for i in range(len(notes_as_int) - sequence_length):
        sequence_in = notes_as_int[i:i + sequence_length]
        network_input.append(sequence_in)

    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1)) / float(vocabulary_size)
    return network_input, assign_int_to_note

def create_midi_from_note_on_off(events, filename):
    song_notes = []
    offset = 0 # Have to implement this because the model struggled with delta time more in results section
    for event in events:
        if event[0] == 1:  # If it's a note on event
            new_note = note.Note(event[1], offset=offset, storedInstrument=instrument.Piano())
            song_notes.append(new_note)
        else:
            song_notes[-1].duration.quarterLength = event[2] - song_notes[-1].offset
            song_notes.append(song_notes[-1])
    
    offset += 0.1
    midi_stream = stream.Stream(song_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))

##############################################################################################

# Loss Function Experiment
# Ctrl + F | Search for loss='binary_crossentropy' and replace with loss=wasserstein_loss
def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

##############################################################################################

def build_generator():
    model = models.Sequential()
    noise = layers.Input(shape=(LATENT_DIMENSION,))
    model = models.Sequential([
        noise,
        layers.Dense(256),
        layers.LeakyReLU(negative_slope=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256),
        layers.LeakyReLU(negative_slope=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512),
        layers.LeakyReLU(negative_slope=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(np.prod(SEQUENCE_SHAPE), activation='sigmoid'),
        layers.Reshape(SEQUENCE_SHAPE)
    ])
    seq = model(noise)
    return models.Model(noise, seq)

def build_discriminator():
    sequence = layers.Input(shape=SEQUENCE_SHAPE)
    model = models.Sequential([
        sequence,
        layers.LSTM(512, return_sequences=True),
        layers.Bidirectional(layers.LSTM(512)),
        layers.Dense(512),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.4),
        layers.Dense(256),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.4),
        layers.Dense(128),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

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

def train(gan, generator, discriminator, network_input, vocabulary_size, assign_int_to_note):
    if os.path.exists("gan_checkpoints"):
        shutil.rmtree("gan_checkpoints")
        os.makedirs("gan_checkpoints")
    else:
        os.makedirs("gan_checkpoints")

    for epoch in range(EPOCHS):
        real_seqs = network_input[np.random.randint(0, network_input.shape[0], BATCH_SIZE)]
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIMENSION))
        fake_seqs = generator.predict(noise, verbose=0)
        d_loss_real = discriminator.train_on_batch(real_seqs, np.ones((BATCH_SIZE, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_seqs, np.zeros((BATCH_SIZE, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIMENSION))
        g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

        print (f"{epoch} Discriminator loss: {d_loss[0]:.5f}, accuracy: {100*d_loss[1]:.2f}% | Generator loss: {g_loss:.5f}")

        if epoch % 10 == 0:
            predictions = generator.predict(noise)
            predictions = [x * (vocabulary_size / 2) + (vocabulary_size / 2) for x in predictions[0]]
            predictions = [assign_int_to_note[int(x[0])] for x in predictions if int(x[0]) in assign_int_to_note]
            if NORMALISTION_METHOD == 0:
                create_midi_chords(predictions, f'gan_checkpoints/generated_song_{epoch}')
            elif NORMALISTION_METHOD == 1:
                create_midi_from_frequencies(predictions, f'gan_checkpoints/generated_song_{epoch}')
            elif NORMALISTION_METHOD == 2:
                create_midi_from_note_on_off(predictions, f'gan_checkpoints/generated_song_{epoch}')


if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    midi_files = glob.glob("../data/classical-piano/Bach_ JS*.mid")        
    
    generator = build_generator()
    discriminator = build_discriminator() 
    gan = build_gan(generator, discriminator)

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002), metrics=['accuracy'])

    if NORMALISTION_METHOD == 0:
        notes = []
        for file in midi_files[:NUMBER_FILES_TO_PARSE]:
            normalised_notes = normalise_song(file)
            notes.extend(normalised_notes)
        
        vocabulary_size = len(set(notes))
        network_input, _, assign_int_to_note = generate_note_sequences(notes, SEQUENCE_LENGTH, vocabulary_size)

        train(gan, generator, discriminator, network_input, vocabulary_size, assign_int_to_note)
        noise = np.random.normal(0, 1, (1, LATENT_DIMENSION))
        predictions = generator.predict(noise)
        predictions = [x * (vocabulary_size / 2) + (vocabulary_size / 2) for x in predictions[0]]
        predictions = [assign_int_to_note[int(x[0])] for x in predictions if int(x[0]) in assign_int_to_note]
        create_midi_chords(predictions, 'generated_song')
        generator.save_weights("final_weights/generator_weights_chords.weights.h5")
        discriminator.save_weights('final_weights/discriminator_weights_chords.weights.h5')
    elif NORMALISTION_METHOD == 1: # Frequency
        notes = []
        for file in midi_files[:NUMBER_FILES_TO_PARSE]:
            normalised_notes = extract_frequencies(file)
            notes.extend(normalised_notes)
        
        vocabulary_size = len(set(notes))
        network_input, _, assign_int_to_note = generate_note_sequences(notes, SEQUENCE_LENGTH, vocabulary_size)
        train(gan, generator, discriminator, network_input, vocabulary_size, assign_int_to_note)
        generator.save_weights("final_weights/generator_weights_freq.weights.h5")
        discriminator.save_weights('final_weights/discriminator_weights_freq.weights.h5')
    elif NORMALISTION_METHOD == 2: # Note Events
        notes = []
        for file in midi_files[:NUMBER_FILES_TO_PARSE]:
            normalised_notes = midi_to_events(file)
            notes.extend(normalised_notes)
        
        vocabulary_size = len(set(tuple(i) for i in notes))
        network_input, assign_int_to_note = generate_note_sequences_events(notes, SEQUENCE_LENGTH, vocabulary_size)
        train(gan, generator, discriminator, network_input, vocabulary_size, assign_int_to_note)
        generator.save_weights("final_weights/generator_weights_events.weights.h5")
        discriminator.save_weights('final_weights/discriminator_weights_events.weights.h5')

