import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

# Reading the spam csv file
data = pd.read_csv('spam_ham_dataset.csv')
data = data[['text', 'label_num']]
#print(data)
x_text = list(data['text'])
y_label = list(data['label_num'])

x_train = x_text[:4137]
x_test = x_text[4137:]
y_train = y_label[:4137]
y_test = y_label[4137:]

num_words = 88000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

# Tokenize our training data
tokenizer = Tokenizer(num_words=num_words, filters='"+,-.<=>[]^_`{|}~\t\n', oov_token=oov_token)
tokenizer.fit_on_texts(x_train)

# Get our training data word index
word_index = tokenizer.word_index

# Encode training data sentences into sequences
train_sequences = tokenizer.texts_to_sequences(x_train)

# Get max training sequence length
maxlen = max([len(x) for x in train_sequences])

# Pad the training sequences
train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

#print("\nTraining sequences:\n", train_sequences)
#print("\nPadded training sequences:\n", train_padded)

test_sequences = tokenizer.texts_to_sequences(x_test)
test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

#print("Testing sequences:\n", test_sequences)
#print("\nPadded testing sequences:\n", test_padded)
#print("\nPadded testing shape:", test_padded.shape)

''' train_padded is used as the training set, while test_padded is used as the testing set '''

'''model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 30))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(30, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_padded[:1000]
x_train = train_padded[1000:]
y_val = y_train[:1000]
y_train1 = y_train[1000:]

model.fit(x_train, y_train1, epochs=30, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_padded, y_test)
print(results)

model = model.save('SpamDetector.h5')'''

model = keras.models.load_model('SpamDetector.h5')
predictions = model.predict(test_padded)

for i in range(20):
    print('\nEmail: \n', x_test[i],  '\n\nPrediction: ', predictions[i], '\n\nActual: ', y_test[i])
