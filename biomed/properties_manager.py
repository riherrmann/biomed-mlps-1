import os as OS
from numpy import float64

class PropertiesManager:
    def __init__(self):
        self.classifier = "is_cancer"
        self.model = "c"
        self.is_blind = False

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
            dtype = float64
        )

        self.selection = dict(
            type = None,
        )

        self.training = dict(
            epochs = 150,
            batch_size = 10,
            validation_split = 0.1,
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
