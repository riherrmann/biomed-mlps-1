from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class TextMiningManager:
    def __init__(self):
        pass

    def train_test_split(self, data):
        training_data, test_data = train_test_split(data, test_size=0.3)
        return training_data, test_data

    def tfidf_transformation(self, training_data, test_data, max_features=200000):
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, max_features=max_features, ngram_range=(1, 4),
                                     sublinear_tf=True)

        vectorizer = vectorizer.fit(training_data['text'])
        training_features = vectorizer.transform(training_data['text'])

        test_features = vectorizer.transform(test_data['text'])
        return training_features, test_features
