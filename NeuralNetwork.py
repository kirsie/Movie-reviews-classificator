import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers


class NeuralNetwork:

    train_input = []
    train_output = []

    test_input = []
    test_output = []

    data_path_dict = {'yelp':   'Data/yelp_labelled.txt',
                      'amazon': 'Data/amazon_cells_labelled.txt',
                      'imdb':   'Data/imdb_labelled.txt'}

    def import_reviews_to_list(self):

        reviews = []
        for source, data_path in self.data_path_dict.items():
            review = pd.read_csv(data_path, names=['sentence', 'label'], sep='\t')
            review['source'] = source  # Add another column filled with the source name
            reviews.append(review)
        reviews = pd.concat(reviews)
        return reviews

    def split_data(self, reviews):

        df_yelp = reviews[reviews['source'] == 'yelp']
        df_amazon = reviews[reviews['source'] == 'amazon']
        df_imdb = reviews[reviews['source'] == 'imdb']

        sentences = df_yelp['sentence'].values
        sentiments = df_yelp['label'].values

        self.train_input, self.test_input, self.train_output, self.test_output = train_test_split(sentences, sentiments)

    def vectorize_string_data(self):

        vectorizer = CountVectorizer()
        vectorizer.fit(self.train_input)

        self.train_input = vectorizer.transform(self.train_input)
        self.test_input = vectorizer.transform(self.test_input)

    def logistic_regression(self):

        classifier = LogisticRegression()
        classifier.fit(self.train_input, self.train_output)

        score = classifier.score(self.test_input, self.test_output)
        print(score)

    def simple_neural_network(self):

        input_dim = self.train_input.shape[1]

        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        history = model.fit(self.train_input, self.train_output,
                            epochs=200, validation_data=(self.test_input, self.test_output),
                            batch_size=300)

        loss, accuracy = model.evaluate(self.train_input, self.train_output, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(self.test_input, self.test_output, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        self.plot_results(history)

    @staticmethod
    def plot_results(history):

        training_accuracy = history.history['acc']
        testing_accuracy = history.history['val_acc']

        training_loss = history.history['loss']
        testing_loss = history.history['val_loss']

        x = range(1, len(training_accuracy)+1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, training_accuracy, 'b', label='Training acc')
        plt.plot(x, testing_accuracy, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, training_loss, 'b', label='Training loss')
        plt.plot(x, testing_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

neural_network = NeuralNetwork()
reviews = neural_network.import_reviews_to_list()
neural_network.split_data(reviews)
neural_network.vectorize_string_data()
neural_network.logistic_regression()
neural_network.simple_neural_network()
