import os as OS

class PropertiesManager:
    def __init__(self):
        self.tfidf_transformation_properties = dict(
            min_df=2,
            max_df=0.95,
            max_features=200000,
            ngram_range=(1, 4),
            sublinear_tf=True,
        )
        self.test_size = 0.3
        self.test_split_size = 0.3
        self.mlp_model_1_properties = dict(
            training_properties=dict(
                epochs=5,
                batch_size=16,
                validation_split=0.1,
            ),
            building_properties=dict(
                loss='categorical_crossentropy',
                optimizer='rmsprop',
                dense_1 = 256,
                activation_1 = 'relu',
                dropout_1 = 0.4,
                dense_2 = 128,
                activation_2 = 'relu',
                dropout_2 = 0.2,
                activation_final = 'softmax',
            ),
            prediction_properties=dict(
                verbose=0,
            )
        )
        self.workers = 2
        self.cache_dir = OS.path.abspath(
            OS.path.join(
                OS.path.dirname( __file__ ), "..", ".cache"
            )
        )
        self.preprocessor_variant = "avn"
