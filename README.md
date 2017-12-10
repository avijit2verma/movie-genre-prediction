# Movie Genre Prediction

This project is for the prediction of the genres of a movie using its plot summary. I have used recurrent neural networks with Long Short Term Memory (LSTM) units for the classification task.

<b>Motivation</b>

Since my childhood, I have been an avid movie watcher. I am generally able to guess the genres of a movie in my mind while reading the plot summary of that movie. So, I got the idea of making this fun little project to see if a deep learning implementation could do the genre prediction task for me.   

At that time, I had been learning the concepts of deep learning and thought this as the perfect way to put them into practice.

<b>Libraries and Frameworks</b>

- Deep Learning library - Keras 2.0 ( with TensorFlow as backend)
- Web Framework - Flask

<b>Dataset</b>

In order to create the dataset for this experiment, you need to download genres.list and plot.list files from ftp://ftp.fu-berlin.de/pub/misc/movies/database/, and then parse files in order to associate the titles, plots, and genres.

I’ve already done this, and parsed both files in order to generate a single file, available here movies_genres.csv, containing the plot and the genres associated to each movie.

<b>Project Description</b>
 
 The user interacts with a web application, currently hosted on localhost, where they can enter the plot summary of the movie and then the trained model makes the prediction about the genres of the movie. I have used the micro web-framework Flask to interface my trained RNN model with the web application.
 
 For the firstpass of the algorithm, the code for the training of the model is invoked. The model is trained using GloVe word embeddings for the encoding of the words. More information about GloVe embeddings can be found <a href = https://nlp.stanford.edu/projects/glove/> here</a>. 

After the firstpass is completed, a file 'firstpass' is created. For any further passes of the algorithm, the code for genre prediction is invoked. The result is passed to the web page through Flask for display.

<b>How to run the program?</b>

In terminal, type : $ python run.py

In web browser, type: http://localhost/lstm/5000/

<b>Results</b>

Due to a lack of time, I haven't been able to compile comprehensive results. 

- The F1 score achieved on the test set by the model in the code was 0.95.
- The exact prediction accuracy achieved on the test set by the model in the code was 78%. 

<b>Future Work</b>

An interesting thought that came into my mind but I wasn't able to implement was to make this a multi-modal classification problem.

We could use, say, the poster of the movie to use visual data along with the already present textual data to improve our classification accuracy. Though, I was not really sure about the mechanics of creating such a dataset and thus could act upon this idea.

If I get the time in the future, I would love to explore the muti-modal classification idea more.

Cheers.
