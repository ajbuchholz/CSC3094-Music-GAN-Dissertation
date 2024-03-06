import glob
from music21 import note, chord, converter, stream, instrument

def encode_midi(midi_file):
    notes = []
    
    midi = converter.parse(midi_file)
    print(f"Parsing {midi_file}")
    notes_to_parse = midi.flatten().notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    
    return notes

def create_midi_chords(prediction_output, filename="output"):
    midi_stream = stream.Stream()
    piano_part = stream.Part()
    piano_part.insert(0, instrument.Piano())
    offset = 0
    song_notes = []

    for item in prediction_output:
        if '.' in item or item.isdigit():
            notes_in_chord = item.split('.')
            notes = [note.Note(int(current_note), offset=offset, storedInstrument=instrument.Piano()) for current_note in notes_in_chord]
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            song_notes.append(new_chord)
        else:
            new_note = note.Note(item, offset=offset, storedInstrument=instrument.Piano())
            song_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(song_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))

    return midi_stream

def process_midi(encoded_midi, max_seq_len=512):
    sequences = []
    for i in range(0, len(encoded_midi), max_seq_len):
        sequence = encoded_midi[i:i+max_seq_len]
        input_seq = ' '.join(str(x) for x in sequence)
        target_seq = "[start] " + ' '.join(str(x) for x in sequence) + " [end]"
        sequences.append((input_seq, target_seq))
    return sequences

def load_midi_file(midi_path, max_seq_len=512):
    encoded_midi = encode_midi(midi_path)
    sequences = process_midi(encoded_midi, max_seq_len=max_seq_len)
    return sequences

def create_midi_data(max_seq_len=512, num_files=None):
    midi_paths = glob.glob("../../data/classical-piano/*.mid", recursive=True)
    if num_files is not None:
        midi_paths = midi_paths[:num_files]

    all_sequences = []

    for path in midi_paths:
        sequences = load_midi_file(path, max_seq_len=max_seq_len)
        all_sequences.extend(sequences)
            
    return all_sequences