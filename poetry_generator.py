from __future__ import absolute_import, division, print_function
import numpy as np
import os
import csv
import tensorflow as tf
import pickle
from pathlib import Path


class PoetryGenerator:

    def __init__(self, create, start_string):

        tf.enable_eager_execution()

        # The maximum length sentence we want for a single input in characters
        seq_length = 100

        text, vocabulary, char2idx, idx2char, text_as_int = self.text_vectorization(create)

        examples_per_epoch = len(text) // seq_length

        # Batch size
        BATCH_SIZE = 64

        steps_per_epoch = examples_per_epoch // BATCH_SIZE

        # Buffer size to shuffle the dataset
        BUFFER_SIZE = 10000

        # Create training dataset
        dataset = self.create_training_data(vocabulary, idx2char, text_as_int)

        # Create training batches
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        # Length of the vocabulary in chars
        vocab_size = len(vocabulary)

        # The embedding dimension
        embedding_dim = 256

        # Number of RNN units
        rnn_units = 1024

        # Check if GPU is available and use it, else use CPU
        if tf.test.is_gpu_available():
            rnn = tf.keras.layers.CuDNNGRU
        else:
            import functools
            rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')

        model = self.build_model(rnn=rnn, vocab_size=len(vocabulary), embedding_dim=embedding_dim,
                                 rnn_units=rnn_units, batch_size=BATCH_SIZE)

        # Directory where the checkpoints will be saved
        here = Path(__file__).parent
        print('######################', here)
        checkpoint_dir = here/'./training_checkpoints'
        print('######################', checkpoint_dir)
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

        if create:
            self.train(checkpoint_callback, model, dataset, steps_per_epoch)
        else:
            tf.train.latest_checkpoint(checkpoint_dir)
            model = self.build_model(rnn, vocab_size, embedding_dim, rnn_units, batch_size=1)
            model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
            model.build(tf.TensorShape([1, None]))
            prediction = self.predict(model, char2idx, idx2char, start_string)
            self.prediction = prediction
            print(prediction)


    def text_vectorization(self, create):

        text = ''
        if create:
            # read your own dataset
            with open('dataset\\PoetryFoundationData.csv', encoding="utf8") as csv_file:

                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0

                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    else:
                        author = row[3]
                        title = row[1]
                        content = row[2]
                        text = text + content
                        line_count += 1

                print(f'Processed {line_count} lines.')

            # The unique characters in the file
            vocabulary = sorted(set(text))
            print(vocabulary)
            pickle.dump(vocabulary, open('resources/Poetry/vocabulary.p', 'wb'))
            print('{} unique characters'.format(len(vocabulary)))

            # Creating a mapping from unique characters to indices
            char2idx = {u: i for i, u in enumerate(vocabulary)}
            print(char2idx)
            pickle.dump(char2idx, open('resources/Poetry/char2idx.p', 'wb'))

            idx2char = np.array(vocabulary)
            pickle.dump(idx2char, open('resources/Poetry/idx2char.p', 'wb'))
            print(idx2char)

            # Convert the whole text to an integer representation using the mapping
            text_as_int = np.array([char2idx[c] for c in text])
            print(text_as_int)
            pickle.dump(text_as_int, open('resources/Poetry/text_as_int.p', 'wb'))

            print('{')
            for char, _ in zip(char2idx, range(20)):
                print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
            print('  ...\n}')

            # Show how the first 13 characters from the text are mapped to integers
            print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

        else:
            vocabulary = pickle.load(open('resources/Poetry/vocabulary.p', 'rb'))
            char2idx = pickle.load(open('resources/Poetry/char2idx.p', 'rb'))
            idx2char = pickle.load(open('resources/Poetry/idx2char.p', 'rb'))
            text_as_int = pickle.load(open('resources/Poetry/text_as_int.p', 'rb'))

        return text, vocabulary, char2idx, idx2char, text_as_int

    def create_training_data(self, vocabulary, idx2char, text_as_int):

        # Training datapoints will consist of text subsets where the target is the input but shifted by one character
        # ex: "Hell" => "ello"
        # The maximum length sentence we want for a single input in characters
        seq_length = 100

        examples_per_epoch = len(vocabulary) // seq_length

        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

        for i in char_dataset.take(5):
            print(idx2char[i])

        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
        for item in sequences.take(5):
            print(repr(''.join(idx2char[item.numpy()])))

        # Generates target and input texts for provided chunk
        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)

        return dataset

    def build_model(self, rnn, vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

    def train(self, checkpoint_callback, model, dataset, steps_per_epoch):

        def loss(labels, logits):
            return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

        model.fit(dataset.repeat(), epochs=10, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

    def predict(self, model, char2idx, idx2char, start_string):

        # Number of characters to generate
        num_generate = 1000

        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 1.0

        # Here batch size == 1
        model.reset_states()

        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

            if (start_string + ''.join(text_generated)).endswith('\n\n\n'):
                return (start_string + ''.join(text_generated)).replace('\n\n', '\n').replace('\t', '').replace('\n\n\n', '\n')

        return (start_string + ''.join(text_generated)).replace('\n\n', '\n').replace('\t', '').replace('\n\n\n', '\n')

    def get_prediction(self):
        return self.prediction


if __name__ == '__main__':
    PoetryGenerator(False, 'The death that brings joy')
