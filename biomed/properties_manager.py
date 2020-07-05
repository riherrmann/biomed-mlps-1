import os as OS

class PropertiesManager:
    def __init__(self):
        self.classifier = "is_cancer"
        self.model = "s"
        self.tfidf_transformation_properties = dict(
            min_df=2,
            max_df=0.95,
            max_features=200000,
            ngram_range=(1, 4),
            sublinear_tf=True,
        )
        self.test_size = 0.3
        self.training_properties = dict(
            epochs = 425,
            batch_size = 10,
            validation_split = 0.1,
            workers = 2,
        )
        self.prediction_properties = dict(
            verbose = 1,
        )
        self.preprocessing = dict(
            workers = 2,
            variant = "lsw",
        )
        self.cache_dir = OS.path.abspath(
            OS.path.join(
                OS.path.dirname( __file__ ), "..", ".cache"
            )
        )

    def __setitem__(self, field, value):
        if not hasattr( self, field ):
            return

        self.__setattr__(field, value)

    def __getitem__(self, field):
        if not hasattr( self, field ):
            return None
        else:
            return self.__getattribute__(field)
