# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from six.moves import cPickle

top_words = 175155
max_words = 500
file_name = '../data/Electronics20000.pkl'
file = open(file_name, 'rb')
vector_length = 100
lstm_neurons = 100
dropout_coeff = 0.5

X_train, y_train, X_test, y_test = cPickle.load(file)

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, vector_length, input_length=max_words, dropout=dropout_coeff))
model.add(Dropout(dropout_coeff))
model.add(LSTM(lstm_neurons))
model.add(Dropout(dropout_coeff))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=200, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
