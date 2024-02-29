import glob
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from utils import normalise_song, generate_note_sequences, create_midi_chords

BATCH_SIZE = 32
NUMBER_FILES_TO_PARSE = 100
NUMBER_OF_EPOCHS = 5
SEQUENCE_LENGTH = 100
GENERATE_NOTES_NUM = 50

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = keras.ops.shape(x)[-1]
        positions = keras.ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def build_model_transformer(vocabulary_size, sequence_length):
    embed_dim = 128  # Increased embedding size
    num_heads = 8  # Increased number of attention heads
    ff_dim = 256  # Increased feed-forward network size
    
    inputs = tf.keras.layers.Input(shape=(sequence_length,))
    embedding_layer = TokenAndPositionEmbedding(sequence_length, vocabulary_size, embed_dim)
    x = embedding_layer(inputs)

    # Adding more Transformer blocks
    for _ in range(4):
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(vocabulary_size, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Custom learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=5000,
        decay_rate=0.75)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_music_transformer(model, start_sequence, assign_int_to_note):
    input_eval = np.array(start_sequence)
    
    prediction_output = []

    for _ in range(GENERATE_NOTES_NUM):
        predictions = model.predict(input_eval)
        predicted_id = np.argmax(predictions[-1])
        prediction_output.append(predicted_id)
        
        input_eval = np.append(input_eval[1:], [[predicted_id]], axis=0)
    
    predicted_notes = [assign_int_to_note[id] for id in prediction_output]
    
    return predicted_notes


if __name__ == '__main__':
    midi_files = glob.glob("../data/classical-piano/*.mid")

    all_notes = []
    for file in midi_files[:NUMBER_FILES_TO_PARSE]:
        normalised_notes = normalise_song(file)
        all_notes.extend(normalised_notes)
    
    vocabulary_size = len(set(all_notes))
    network_input, network_output, assign_int_to_note = generate_note_sequences(notes=all_notes, sequence_length=SEQUENCE_LENGTH, vocabulary_size=vocabulary_size)
    model = build_model_transformer(vocabulary_size, SEQUENCE_LENGTH)
    model.fit(network_input, network_output, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE)
    start_sequence = network_input[0]
    prediction_output = generate_music_transformer(model, start_sequence, assign_int_to_note)
    create_midi_chords(prediction_output, file_path="transformer_ouput.mid")
