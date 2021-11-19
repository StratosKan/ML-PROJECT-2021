from bs4 import BeautifulSoup

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

sentence = sentences[0]
soup = BeautifulSoup(sentence, features="html.parser")
sentence = soup.get_text()

stopwords = ["a", "about", "above", "yours", "is", "yourself", "yourselves"]

words = sentence.split()
filtered_sentence = ""
for word in words:
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "

sentences.append(filtered_sentence)
print(sentences)

import string
sentence = sentences[0]
table = str.maketrans('', '', string.punctuation)
words = sentence.split()
filtered_sentence = ""
for word in words:
    word = word.translate(table)
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "
sentences.append(filtered_sentence)
print(sentences)
