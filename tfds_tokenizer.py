import tensorflow_datasets as tfds
import tensorflow as tf
import string
from bs4 import BeautifulSoup

stopwords = ["a", "is", "you", "yourselves"]

table = str.maketrans('', '', string.punctuation)

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
for item in train_data:
    sentence = str(item['text'].decode('UTF-8').lower())
    soup = BeautifulSoup(sentence, features="html.parser")
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    imdb_sentences.append(filtered_sentence)
    # imdb_sentences.append(str(item['text']))

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)

# print(tokenizer.word_index)
# print(len(tokenizer.word_index))

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

# DECODER [642, 5260, 257] = today sunny day
reverse_word_index = dict(
    [(value, key) for (key, value) in tokenizer.word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in sequences[0]])
print(decoded_review)

# imdb subwords dataset

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True
)
encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))  # 8185
# print(encoder.subwords)

# ENCODING/DECODING HARDCODED EXAMPLES
sample_string = 'Today is a sunny day'
encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
test_string = encoder.decode([6427, 4869, 9, 4, 2365, 1361, 606])
print(test_string)
print(encoder.subwords[6426])  # Tod?
