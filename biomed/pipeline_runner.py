from biomed.pipeline import Pipeline

class PipelineRunner:
    class Factory:
        @staticmethod
        def getInstance():
            return PipelineRunner()

    def run( self, Permutations: list ):
        Pipe = Pipeline.Factory.getInstance()
        for Configuration in Permutations:
            self.__runPipeline( Pipe, Configuration )

    def __runPipeline( self, Pipe: Pipeline, Configuration: dict ):
        Pipe.pipe(
            Data = Configuration[ "trainings_data" ],
            TestData = Configuration[ "test_data" ],
            ShortName = Configuration[ "shortname" ],
            Description = Configuration[ "description" ],
            Properties = Configuration
        )
