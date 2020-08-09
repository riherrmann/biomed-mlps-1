from numpy import array as Array

class InputData:
    def __init__( self, Training: Array, Validation: Array, Test: Array = None ):
        self.Training = Training
        self.Validation = Validation
        self.Test = Test

    def __str__( self ):
        return "(<Training>{}, <Validation>{}, <Test>{})".format(
            str( self.Training ),
            str( self.Validation ),
            str( self.Test )
        )
