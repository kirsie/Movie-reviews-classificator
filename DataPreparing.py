import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class Data:

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

    def split__data(self, reviews):

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


data = Data()
reviews = data.import_reviews_to_list()
data.split__data(reviews)
data.vectorize_string_data()
data.logistic_regression()