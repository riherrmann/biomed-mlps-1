from biomed.pipeline import Pipeline

class PipelineRunner:
    class Factory:
        @staticmethod
        def getInstance():
            return PipelineRunner()

    def __init__( self ):
        self.__Output = None

    def run( self, Permutations: list ):
        self.__Output = {}
        self.__runPipeline( Permutations )
        return dict( self.__Output )

    def __runPipeline( self, Permutations: list ):
        Pipe = Pipeline.Factory.getInstance()
        for Configuration in Permutations:
            self.__Output[ Configuration[ "id" ] ] = Pipe.pipe(
                Configuration[ "training" ],
                Configuration[ "test" ],
                Configuration
            )
