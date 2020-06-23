from biomed.pipeline import pipeline
from math import ceil
from multiprocessing import Process
from time import sleep

class PipelineRunner:
    def __getChunkSize(
        self,
        SizeOfPermutations: int,
        TotalProcesses: int
    ) -> int:
        return ceil( SizeOfPermutations / TotalProcesses )

    def __computeChunk( self, N: int, BagOfStuff: list ) -> list:
        for Index in range( 0, len( BagOfStuff ), N ):
            yield BagOfStuff[ Index:Index + N]

    def run( self, Permutations: list, Workers: int ):
        Jobs = list()
        DistributedPermuation = self.__computeChunk(
            self.__getChunkSize( len( Permutations ), Workers ),
            Permutations
        )

        self.__spawn( Jobs, DistributedPermuation )
        sleep( 0 )# yield
        self.__waitUntilDone( Jobs )


    def __spawn( self, Jobs: list, Chunks: list ):
        for Chunk in Chunks:
            Job = Process(
                target = PipelineRunner.__run,
                args = ( self, Chunk )
            )

            Job.start()
            Jobs.append( Job )

    def __waitUntilDone( self, Jobs ):
        for Job in Jobs:
            Job.join()

    @staticmethod
    def __run( This, Permutations: list ):
        pass
