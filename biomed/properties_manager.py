import os as OS

class PropertiesManager:
    def __init__(self):
        self.classifier = "is_cancer"
        self.model = "c"

        self.splitting = dict(
            folds = 1,
            test = 0.2,
            validation = 0.2,
            seed = 1
        )

        self.preprocessing = dict(
            workers = 90,
            variant = "lanv",
        )

        self.vectorizing = dict(
            min_df = 2,
            max_df = 0.95,
            max_features = 200000,
            ngram_range = ( 1, 4 ),
            sublinear_tf = True,
            binary = False,
            norm =  'l2',
            analyzer = 'word', #{‘word’, ‘char’, ‘char_wb’}
            use_idf = True,
            smooth_idf = True,
        )

        self.selection = dict(
            type = False,
        )

        self.training = dict(
            epochs = 150,
            batch_size = 10,
            workers = 90,
        )

        self.predictions = dict(
            verbose = 1,
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
