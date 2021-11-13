import tensorflow as tf
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers

'''stockdata = pd.read_csv('stockmarket.csv')
closestockdate = list(stockdata['Date'])
datanews = pd.read_csv('RedditNews.csv')['News']
datanews2 = pd.read_csv('RedditNews.csv')['Date']
datanews3 = pd.read_csv('RedditNews.csv')
datanews = [x.replace('\r\n', '') for x in datanews]
#print(datanews)
Newsdata = datanews
Date = list(datanews2)
Newsdata1 = Newsdata[::-1]
Date1 = Date[::-1]
data_shit = list(zip(Date1, Newsdata1))
dates = (list((datanews3.Date.unique())))
dates1 = dates[::-1]
Newsdates2 = []
Newsdates3 = []
for k in range(0, len(dates1)):
    Newsdates2 += [[dates1[k]]]
#print(dates2)

for i in range(0, len(dates1)):
    for j in range(0, len(data_shit)):
        if dates1[i] == data_shit[j][0]:
            Newsdates2[i] += [data_shit[j][1]]
#print(len(Newsdates2))

for m in range(0, len(Newsdates2)):
    for n in range(0, len(closestockdate)):
        if Newsdates2[m][0] == closestockdate[n]:
            Newsdates3.append(Newsdates2[m])

print(Newsdates3[900])
#print(len(closestockdate))


header = ['NewsDates']

with open('NewsDates.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    # write the header
    writer.writerow(header)
    # write multiple rows
    writer.writerows(Newsdates3)'''

dataset = pd.read_csv('NewsDates.csv', sep='delimiter')['NewsDates']
#dataset.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
dataset2 = list(dataset)
test_dataset = dataset2[1611:]
train_dataset = dataset2[:1611]
'''countvector = CountVectorizer(ngram_range=(2, 2))
traindataset = countvector.fit_transform(train_dataset)
testdataset = countvector.transform(test_dataset)
#print(traindataset)'''

num_words = 100000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

# Tokenize our training data
tokenizer = Tokenizer(num_words=num_words, filters='0123456789"/+,()-.<=>[]^_`{|}~\t\n', oov_token=oov_token)
tokenizer.fit_on_texts(train_dataset)

# Get our training data word index
word_index = tokenizer.word_index

# Encode training data sentences into sequences
train_sequences = tokenizer.texts_to_sequences(train_dataset)

# Get max training sequence length
maxlen = max([len(x) for x in train_sequences])

# Pad the training sequences
train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
#x_val = train_padded[:300]
#x_train = train_padded[300:]

# Encoding our test data
test_sequences = tokenizer.texts_to_sequences(test_dataset)
test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

#print('Test padded:\n', test_padded)
#print('\nTrain padded:\n', train_padded)

# Labels processing
stockdata = pd.read_csv('stockmarket.csv')
closestock = list(stockdata['Close'])[::-1]
#print(closestock)

# Get our labels, 0 for decrease in stock prize and 1 for increase/constant stock prize
Label = [0]
for i in range(0, len(closestock)-1):
    if closestock[i+1] >= closestock[i]:
        Label.append(1)
    elif closestock[i+1] < closestock[i]:
        Label.append(0)

label_train = Label[:1611]
label_test = Label[1611:]
#y_val = label_train[:300]
#y_train = label_train[300:]

# Define the model

model = keras.Sequential()
model.add(keras.layers.Embedding(100000, 32))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=regularizers.l2(5e-4), bias_regularizer=regularizers.l2(5e-4)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.00001, decay=1e-6)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_padded, label_train, epochs=10, verbose=1)
results = model.evaluate(test_padded, label_test)
print(results)

#model = model.save('NewsStock.h5')

predictions = model.predict(test_padded)

for i in range(10):
    print('\nPrediction: ', predictions[i], '\n\nActual: ', label_test[i])

'''model = RandomForestClassifier(n_estimators=200, criterion='entropy')
model.fit(traindataset, label_train)
acc = model.score(testdataset, label_test)
print((acc))
predictions = model.predict(testdataset)

for i in range(10):
    print('\nPrediction: ', predictions[i], '\n\nActual: ', label_test[i])'''













