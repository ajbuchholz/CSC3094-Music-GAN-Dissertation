import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import shutil
import glob
import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
from keras import optimizers
from utils import create_midi_chords
from gan_model import build_generator, build_discriminator, build_gan, generate_note_sequences

NUMBER_FILES_TO_PARSE = 200
EPOCHS = 250
BATCH_SIZE = 128
SEQUENCE_LENGTH = 100
SEQUENCE_SHAPE = (SEQUENCE_LENGTH, 1)
LATENT_DIMENSION = 1000
NUMBER_OF_NOTES = 100

def normalise_song_multi_track(midi_file):
    print(f"Parsing {midi_file}")
    midi = converter.parse(midi_file)
    
    instrument_notes = {}
    for part in midi.parts:
        # Attempt to retrieve the instrument name, default to 'Piano' if not found
        instr_name = 'Piano'
        for el in part.recurse():

            if isinstance(el, instrument.Instrument):
                if 'Piano' in el.instrumentName:
                    instr_name = 'Acoustic Grand Piano'
                else:
                    instr_name = el.instrumentName
                break 
        
        notes_to_parse = part.recurse().notes
        notes = [] 

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        if instr_name in instrument_notes:
            instrument_notes[instr_name].extend(notes)
        else:
            instrument_notes[instr_name] = notes 

    return instrument_notes

def process_midi_files_multi_track(midi_files):
    all_instruments = {}

    for midi_file in midi_files:
        instrument_notes = normalise_song_multi_track(midi_file)
        
        for instr, notes in instrument_notes.items():
            if instr in all_instruments:
                all_instruments[instr].extend(notes)
            else:
                all_instruments[instr] = notes

    return all_instruments

def create_midi_file_multi_track(piano_notes, violin_notes, output_file):
    piano_array = []
    violin_array = []
    offset = 0

    # Piano part
    for item in piano_notes:
        if '.' in item or item.isdigit():
            notes_in_chord = item.split('.')
            notes = [note.Note(int(current_note), offset=offset) for current_note in notes_in_chord]
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            piano_array.append(new_chord)
        else:
            new_note = note.Note(item, offset=offset)
            piano_array.append(new_note)
        offset += 1

    # Violin part
    offset = 0
    for item in violin_notes:
        if '.' in item or item.isdigit():
            notes_in_chord = item.split('.')
            notes = [note.Note(int(current_note), offset=offset) for current_note in notes_in_chord]
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            violin_array.append(new_chord)
        else:
            new_note = note.Note(item, offset=offset)
            violin_array.append(new_note)
        offset += 1

    piano_stream = stream.Part(piano_array)
    piano_stream.insert(0, instrument.Piano())
    violin_stream = stream.Part(violin_array)
    violin_stream.insert(0, instrument.Violin())
    midi_stream = stream.Stream()
    midi_stream.append(piano_stream)
    midi_stream.append(violin_stream)
    midi_stream.write('midi', fp=output_file)

def train(gan, generator, discriminator, network_input, vocabulary_size, assign_int_to_note, instrument):
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
            create_midi_chords(predictions, f"gan_checkpoints/generated_song_{epoch}_{instrument}", instrument)

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    midi_files = glob.glob("../data/mozart-multi/*.mid")        
    
    generator_piano = build_generator()
    discriminator_piano = build_discriminator() 
    gan_piano = build_gan(generator_piano, discriminator_piano)
    discriminator_piano.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002), metrics=['accuracy'])

    generator_strings = build_generator()
    discriminator_strings = build_discriminator() 
    gan_strings = build_gan(generator_strings, discriminator_strings)
    discriminator_strings.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002), metrics=['accuracy'])

    notes = process_midi_files_multi_track(midi_files)
    
    vocabulary_size_piano = len(set(notes["Acoustic Grand Piano"]))
    vocabulary_size_strings = len(set(notes["StringInstrument"]))

    network_input_piano, _, assign_int_to_note_piano = generate_note_sequences(notes["Acoustic Grand Piano"], SEQUENCE_LENGTH, vocabulary_size_piano)
    network_input_strings, _, assign_int_to_note_strings = generate_note_sequences(notes["StringInstrument"], SEQUENCE_LENGTH, vocabulary_size_strings)

    if os.path.exists("gan_checkpoints"):
        shutil.rmtree("gan_checkpoints")
        os.makedirs("gan_checkpoints")
    else:
        os.makedirs("gan_checkpoints")
    
    train(gan_piano, generator_piano, discriminator_piano, network_input_piano, vocabulary_size_piano, assign_int_to_note_piano, 'Acoustic Grand Piano')
    noise = np.random.normal(0, 1, (1, LATENT_DIMENSION))
    predictions_piano = generator_piano.predict(noise)
    predictions_piano = [x * (vocabulary_size_piano / 2) + (vocabulary_size_piano / 2) for x in predictions_piano[0]]
    predictions_piano = [assign_int_to_note_piano[int(x[0])] for x in predictions_piano if int(x[0]) in assign_int_to_note_piano]

    train(gan_strings, generator_strings, discriminator_strings, network_input_strings, vocabulary_size_strings, assign_int_to_note_strings, 'StringInstrument')
    noise = np.random.normal(0, 1, (1, LATENT_DIMENSION))
    predictions_strings= generator_strings.predict(noise)
    predictions_strings = [x * (vocabulary_size_strings / 2) + (vocabulary_size_strings / 2) for x in predictions_strings[0]]
    predictions_strings = [assign_int_to_note_strings[int(x[0])] for x in predictions_strings if int(x[0]) in assign_int_to_note_strings]

    create_midi_file_multi_track(predictions_piano, predictions_strings, 'multitrack_song.mid')