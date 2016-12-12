# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from six.moves import cPickle

top_words = 175155
max_words = 500
file_name = '../Electronics20000.pkl'
file = open(file_name, 'rb')
vector_length = 100
perceptrons_layer1 = 100
perceptrons_layer2 = 1

X_train, y_train, X_test, y_test = cPickle.load(file)

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, vector_length, input_length=max_words))
model.add(Flatten())
model.add(Dense(perceptrons_layer1, activation='relu'))
model.add(Dense(perceptrons_layer2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
