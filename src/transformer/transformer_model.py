# Modified From https://keras.io/examples/nlp/neural_machine_translation_with_transformer/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import tensorflow as tf
from keras.layers import TextVectorization
from transformer_utils import create_midi_data, create_midi_chords
import keras
import keras.ops as ops
from keras import layers

VOCABULARY_SIZE = 15000
SEQUENCE_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 10
EMBED_DIMS = 256
LATENT_DIMS = 2048
NUM_HEADS = 8


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
            padding_mask = ops.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = ops.arange(sequence_length)[:, None]
        j = ops.arange(sequence_length)
        mask = ops.cast(i >= j, dtype="int32")
        mask = ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = ops.concatenate(
            [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])],
            axis=0,
        )
        return ops.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


################################################################################################################

def format_dataset(inp, tar):
    inp = input_vectorization(inp)
    tar = target_vectorization(tar)
    return (
        {
            "encoder_inputs": inp,
            "decoder_inputs": tar[:, :-1],
        },
        tar[:, 1:],
    )

def make_dataset(pairs):
    inp_texts, tar_texts = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(inp_texts), list(tar_texts)))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)

def decode_sequence(input_sentence):
    tokenized_input_sentence = input_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = ops.convert_to_numpy(
            ops.argmax(predictions[0, i, :])
        ).item(0)
        sampled_token = tar_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence

def custom_standardization(input_text):
    return tf.strings.regex_replace(input_text, "[^a-zA-Z0-9.-]", " ")

################################################################################################################

sequences = create_midi_data(max_seq_len=100)
random.shuffle(sequences)
num_val_samples = int(0.15 * len(sequences))
num_train_samples = len(sequences) - 2 * num_val_samples
train_pairs = sequences[:num_train_samples]
val_pairs = sequences[num_train_samples : num_train_samples + num_val_samples]
test_pairs = sequences[num_train_samples + num_val_samples :]

print(f"{len(sequences)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

input_vectorization = TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize=custom_standardization
)
target_vectorization = TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH + 1,
    standardize=custom_standardization
)
train_inp_texts = [pair[0] for pair in train_pairs]
train_tar_texts = [pair[1] for pair in train_pairs]
input_vectorization.adapt(train_inp_texts)
target_vectorization.adapt(train_tar_texts)

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(SEQUENCE_LENGTH, VOCABULARY_SIZE, EMBED_DIMS)(encoder_inputs)
encoder_outputs = TransformerEncoder(EMBED_DIMS, LATENT_DIMS, NUM_HEADS)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIMS), name="decoder_state_inputs")
x = PositionalEmbedding(SEQUENCE_LENGTH, VOCABULARY_SIZE, EMBED_DIMS)(decoder_inputs)
x = TransformerDecoder(EMBED_DIMS, LATENT_DIMS, NUM_HEADS)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(VOCABULARY_SIZE, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")

transformer.summary()
transformer.compile("rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
transformer.save_weights("../final_weights/transformer_weights.weights.h5")

# transformer.load_weights("../final_weights/transformer_weights.weights.h5")

# Generate music
tar_vocab = target_vectorization.get_vocabulary()
tar_index_lookup = dict(zip(range(len(tar_vocab)), tar_vocab))
max_decoded_sentence_length = 100

test_input_seqeuences = [pair[0] for pair in test_pairs]
input_sequence = random.choice(test_input_seqeuences)
song_sequence = decode_sequence(input_sequence)
song_array = song_sequence.replace("[start]", "").replace("end", "").split()
create_midi_chords(song_array, filename="transformer_output")
