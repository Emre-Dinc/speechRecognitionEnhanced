import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Load the dataset
# The dataset should consist of Turkish words along with their morphological analyses
# Each word and its analysis should be separated by a tab character
# For example: "kitap\t[Noun+A3sg+Pnon+Nom]"
with open("dataset.txt", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

words = []
analyses = []
for line in lines:
    parts = line.split("\t")
    word = parts[0]
    analysis = parts[1]
    words.append(word)
    analyses.append(analysis)

# Prepare the input and output data
# Convert each character in the words to a numerical value
# Convert each analysis to a sequence of numerical values
# The input sequence length is the maximum length of the words in the dataset
# The output sequence length is the maximum length of the analyses in the dataset
word_maxlen = max(len(w) for w in words)
analysis_maxlen = max(len(a) for a in analyses)

word_chars = sorted(set("".join(words)))
analysis_chars = sorted(set("".join(analyses)))

word_char2index = dict((c, i) for i, c in enumerate(word_chars))
analysis_char2index = dict((c, i) for i, c in enumerate(analysis_chars))

X = np.zeros((len(words), word_maxlen, len(word_chars)), dtype=np.float32)
Y = np.zeros((len(analyses), analysis_maxlen, len(analysis_chars)), dtype=np.float32)

for i, word in enumerate(words):
    for j, char in enumerate(word):
        X[i, j, word_char2index[char]] = 1
for i, analysis in enumerate(analyses):
    for j, char in enumerate(analysis):
        Y[i, j, analysis_char2index[char]] = 1

# Define the LSTM model
model = keras.Sequential()
model.add(layers.LSTM(128, input_shape=(word_maxlen, len(word_chars)), return_sequences=True))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(analysis_chars), activation="softmax")))

model.compile(loss="categorical_crossentropy", optimizer="adam")

# Train the model
model.fit(X, Y, batch_size=128, epochs=50)

# Use the model for predictions
# You can input any Turkish word to the model and get its morphological analysis as output
word = "kitap"
x = np.zeros((1, word_maxlen, len(word_chars)))
for i, char in enumerate(word):
    x[0, i, word_char2index[char]] = 1

y = model.predict(x)[0]
analysis = ""
for i in range(analysis_maxlen):
    index = np.argmax(y[i])
    char = analysis_chars[index]
    if char == "\n":
        break
    analysis += char

print(f"Word: {word}")
print(f"Analysis: {analysis}")