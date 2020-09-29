import os as OS

class PropertiesManager:
    def __init__(self):
        self.classifier = "is_cancer"
        self.model = "b2"

        self.splitting = dict(
            folds = 1,
            test = 0.2,
            validation = 0.2,
            seed = 1
        )

        self.preprocessing = dict(
            workers = 1,
            variant = "lanv",
        )

        self.vectorizing = dict(
            min_df = 1,
            max_df = 0.5,
            max_features = 100000,
            ngram_range = ( 1, 2 ),
            sublinear_tf = True,
            binary = False,
            norm =  'l2',
            analyzer = 'word', #{‘word’, ‘char’, ‘char_wb’}
            use_idf = True,
            smooth_idf = True,
        )

        self.selection = dict(
            type = None,
            amountOfFeatures = 500,
            treeEstimators = 250,
            treeMaxFeatures = 500,
        )

        self.training = dict(
            epochs = 150,
            batch_size = 10,
            workers = 1,
            patience = 100,
        )

        self.weights = dict(
            use_class_weights = False
        )

        self.predictions = dict(
            verbose = 1,
        )

        self.evaluator = dict(
            captureFeatures = False
        )

        self.cache_dir = OS.path.abspath(
            OS.path.join(
                OS.path.dirname( __file__ ), "..", ".cache"
            )
        )

        self.result_dir = OS.path.abspath(
            OS.path.join(
                OS.path.dirname( __file__ ), "..", "results"
            )
        )

    def toDict( self ) -> dict:
        ShallowCopy = dict()
        for Key in self.__dict__.keys():
            ShallowCopy[ Key ] = self[ Key ]

        return ShallowCopy

    def __setitem__( self, field, value ):
        if not hasattr( self, field ):
            return

        self.__setattr__( field, value )

    def __getitem__( self, field ):
        if not hasattr( self, field ):
            return None
        else:
            return self.__getattribute__(field)
