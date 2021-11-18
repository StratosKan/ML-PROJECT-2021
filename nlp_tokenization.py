import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)

test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)

# PADDING TIME

from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences.append("I really enjoyed walking in the snow today")
print(sentences)

sequences = tokenizer.texts_to_sequences(sentences)  # better do pre-padding
padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post')  # or use tf.RaggedTensor
print(padded)
