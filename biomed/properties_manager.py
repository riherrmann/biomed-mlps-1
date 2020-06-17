class PropertiesManager:
    def __init__(self):
        self.tfidf_transformation_properties = dict(
            min_df=2,
            max_df=0.95,
            max_features=200000,
            ngram_range=(1, 4),
            sublinear_tf=True
        )
        self.test_size = 0.3
