 # Multi-Label Movie Genre Classification from the movie plot summary

import json,os
from flask import Flask, jsonify, request, render_template
import numpy as np
import keras
import keras.preprocessing.text
import nltk
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras import optimizers

from nltk.corpus import stopwords

import pandas

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
np.random.seed(7)  # for reproducibility

@app.route('/lstm')
def my_form():
	return render_template("movie_genre.html")



@app.route('/lstm', methods=['POST'])
def model_and_predict():
	plots = request.form['plots']
	#print (plots)
	if not os.path.exists('firstpass'):
	    os.mknod('firstpass')
		
		print "Loading already processed training data"
		data_df = pd.read_csv("movies_genres.csv", delimiter='\t')
		# all the list of genres to be used by the classification report
		genres = list(data_df.drop(['title', 'plot', 'plot_lang'], axis=1).columns.values)


	    print (len(code))
	    x_list = data_x = data_df[['plot']].values.tolist()      #creating a list of plots
	    y_list = genres										    #corresponding list of genres	

	    for k in range(len(x_list)):
	        wor = x_list[k].split()
	        filtered_words = [w for w in wor if w not in stopwords.words('english')]
	        x_list[k] = filtered_words
	        str1= ' '.join(x_list[k])
	        x_list[k] = str1

	    print (type(x_list))
	    #convert list to numpy.ndarray
	    x = np.array(x_list) #indvidual element of the array is a string
	    y = np.array(y_list) #indvidual element of the array is a string
	    #__________________________________________________
	    # preprocessing the data obtained

	    tk = keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
	    tk.fit_on_texts(x)

	    x_train = tk.texts_to_sequences(x)

	    word_index = tk.word_index  # index of unique words
	    print('Found %s unique tokens.' % len(word_index))



	    mlb = MultiLabelBinarizer()
	    y_train  = mlb.fit_transform(y)

	    #____________________________________________________
	    # setting up the parameters and the data for the model

	    max_len = 500       #length of sequence
	    batch_size = 256
	    epochs = 500
	    max_features = len(word_index) + 1   # (number of words in the vocabulary) + 1

	    x_train = sequence.pad_sequences(x_train, maxlen=max_len, padding='pre')

	    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)



	    print('x_train shape:', x_train.shape)
	    print('x_test shape:', x_test.shape)

	    print('y_train shape:', y_train.shape)
	    print('y_test shape:', y_test.shape)
	    label_num = len(y_train[0])
	    #________________________________________________________
	    # preparing the embedding layer using GloVe embeddings
	    
	    embeddings_index = {}
	    f = open('glove.6B.100d.txt')
	    for line in f:
	        values = line.split()
	        word = values[0]
	        coefs = np.asarray(values[1:], dtype='float32')
	        embeddings_index[word] = coefs
	    f.close()

	    print('Found %s word vectors.' % len(embeddings_index))

	    embedding_matrix = np.zeros((max_features, 100))

	    for word, i in word_index.items():
	        embedding_vector = embeddings_index.get(word)
	        if embedding_vector is not None:
	            embedding_matrix[i] = embedding_vector
	    #words not found in embedding index will be all zeros
	    embedding_layer = Embedding(input_dim = max_features,
	                                output_dim = 100,
	                                weights=[embedding_matrix],
	                                mask_zero = True,
	                                input_length = max_len,
	                                trainable = False)

	    
	    #________________________________________________________
	    # building the model and compiling it


	    model = Sequential()
	    #model.add(Embedding(input_dim = max_features, output_dim = 64, mask_zero = True, input_length = max_len))
	    model.add(embedding_layer)
	    model.add(Dropout(0.3))
	    model.add(LSTM(64, return_sequences = True))
	    model.add(Dropout(0.3))
	    model.add(LSTM(64))
	    model.add(Dropout(0.3))
	    model.add(Dense(label_num, activation = 'sigmoid'))

	    print(model.summary())

	    rmsprop = optimizers.RMSprop(lr = 0.01, decay = 0.0001)
	    model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics=['accuracy'])


	    #________________________________________________________
	    # saving the model to a json file



	    #json_string = model.to_json()

	    #with open("model.json","w") as text_file:
	        #text_file.write(json_string)

	    #________________________________________________________
	    # fitting the data on the model and saving the weights obtained

	    model.fit(x_train, y_train, epochs=epochs, batch_size= batch_size,validation_split=0.2)

	    #model.save_weights('my_model_weights.h5')
	    model.save('my_model.h5')

	    out = model.predict(x_test)
	    out = np.array(out)
	    print out

	    y_pred = np.zeros(out.shape)

	    #for i in range(0, len(out[:,0])):
	    #    a = out[i,:]
	    #    index = np.argpartition(a,-5)[-5:]
	    #    for j in range(len(index)):
	    #        y_pred[i, index[j]] = 1

	    y_pred[out>0.5]=1
	    y_pred = np.array(y_pred)




	    # Calculation of the performance metrics when we have a separate validation set

	    print '-------------------------------------------------------------'
	    #print xt[0:1]
	    print '-------------------------------------------------------------'
	    #print y_pred
	    print '-------------------------------------------------------------'
	    #print y_test
	    print '-------------------------------------------------------------'
	    hl = hamming_loss(y_test,y_pred)
	    score = accuracy_score(y_test, y_pred)
	    precision = metrics.precision_score(y_test,y_pred, average = 'samples')
	    recall = metrics.recall_score(y_test,y_pred, average = 'samples')
	    f1 = metrics.f1_score(y_test,y_pred, average = 'samples')

	    print "Hamming loss:", hl
	    print "score:", score
	    print "Precision:", precision
	    print "Recall:", recall
	    print "F1_score:", f1

	    joblib.dump(y_list, 'y_list')

	else:

		#with open("model.json",'r') as content_file:
	        #json_string = content_file.read()

	    #model = model_from_json(json_string)
	    #model.load_weights('my_model_weights.h5')
	    model = load_model('my_model.h5')
	    print '-----------------------------'
	    print 'Loading the model...'
	    print '-----------------------------'
        y_list = joblib.load('y_list')
	    y = np.array(y_list)
	    mlb = MultiLabelBinarizer()
	    y_train  = mlb.fit_transform(y)
	    max_len = 500       #length of sequence
	    batch_size = 1024

	    plots = str(plots)

	    xt = [plots]

	    for k in range(len(xt)):
	        wor = xt[k].split()
	        filtered_words = [w for w in wor if w not in stopwords.words('english')]
	        xt[k] = filtered_words
	        str1= ' '.join(xt[k])
	        xt[k] = str1
            print '-------------------------------'
	    print 'Processing the text...'
            print '-------------------------------'
	    tk_xt = keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
	    tk_xt.fit_on_texts(xt)

	    xt = tk_xt.texts_to_sequences(xt)
	    #print xt
	    xt = sequence.pad_sequences(xt, maxlen=max_len, padding='pre')
	    #print xt
	    xt = np.array(xt)

	    out = model.predict(xt[0:1])
	    out = np.array(out)
            print '--------------------------------------------------------------------------'
            print 'Predicting the probabilities of genres according to the movie plot...'
	    print (out)
	    print '--------------------------------------------------------------------------'
	    y_pred = np.zeros(out.shape)

	    #for i in range(0, len(out[:,0])):
	        #a = out[i,:]
	        #index = np.argpartition(a,-5)[-5:]
	        #for j in range(len(index)):
	            #y_pred[i, index[j]] = 1

	    #y_pred[out>0.5]=1
	    y_pred = np.array(y_pred)


	    #_____________________________________________________________________________________

	    #code to select threshold for each label using Matthew's Correlation Coefficient (MCC)
	    
		# This gives each label a threshold probability, above which it is included in prediction and below which it is not included in prediction.
	    
	    threshold = np.arange(0.1,0.9,0.1)

	    acc = []
	    accuracies = []
	    best_threshold = np.zeros(out.shape[1])
	    for i in range(out.shape[1]):
	        y_prob = np.array(out[:,i])
	        for j in threshold:
	            y_pred = [1 if prob>=j else 0 for prob in y_prob]
	            y_pred = np.array(y_pred)
	            print '_________________________________________________'
	            print y_test[:,i]
	            print '_________________________________________________'
	            print y_pred
	            print '_________________________________________________'
	            mcc = matthews_corrcoef(y_test[:,i],y_pred)

	            acc.append(mcc)
	        acc   = np.array(acc)
	        index = np.where(acc==acc.max())
	        accuracies.append(acc.max())
	        best_threshold[i] = threshold[index[0][0]]
	        acc = []

	    print best_threshold

	    y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
	   #_______________________________________________________________________________________________

	    #_________________________________________________________

	    # getting the label names back from the binarizer for an example

	    #predicted = loaded_model.predict(X_test)

	    all_labels = mlb.inverse_transform(y_pred)
	    result = []
	    for item, labels in zip(xt, all_labels):
	        result.append( ', '.join(labels))

	    print(plots)
	    print(result)
		
		return render_template("movie_genre.html",result=result, plots=plots)