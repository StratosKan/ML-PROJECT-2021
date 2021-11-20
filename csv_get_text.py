import csv
import os.path
import string
from bs4 import BeautifulSoup
import tensorflow as tf

sentences = []
labels = []
stopwords = ["a", "is", "you", "yourselves"]
table = str.maketrans('', '', string.punctuation)

project_path = os.path.dirname(os.path.abspath(__file__))
print(project_path)
csv_path = project_path + '/tmp/text_emotion.csv'
print(csv_path)
with open(csv_path, encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader)  # skip 1st line
    for row in reader:
        labels.append(int(row[0]))
        sentence = row[1].lower()
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        soup = BeautifulSoup(sentence, features="html.parser")
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = ""
        for word in words:
            word = word.translate(table)
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + " "
        sentences.append(filtered_sentence)

print(len(sentences))
print(len(labels))
print(sentences[39000])
print(labels[39000])

# Preparing dataset for training
training_size = 28000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

vocab_size = 20000
max_length = 10
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)  # required before texts_to_sequences

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = tf.keras.preprocessing.sequence.pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(training_sequences[20000])
print(training_padded[20000])
print(word_index)
