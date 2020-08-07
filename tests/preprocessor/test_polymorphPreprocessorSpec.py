import unittest
from unittest.mock import MagicMock
from biomed.preprocessor.normalizer.normalizer import Normalizer
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.preprocessor import Preprocessor
from biomed.preprocessor.polymorph_preprocessor import PolymorphPreprocessor
from biomed.properties_manager import PropertiesManager
from pandas import Series
from multiprocessing import Manager

class StubbedNormalizer( Normalizer ):
    def __init__( self ):
        self.WasCalled = False
        self.CallCounter = 0

    def apply( self, Text: str, Flags: str ) -> list:
        self.WasCalled = True
        self.CallCounter += 1
        return Text

class StubbedNormalizerFactory( NormalizerFactory ):
    def __init__( self, GivenFlags: list ):
        self.CallCounter = 0
        self.LastNormalizers = list()
        self.__Flags = GivenFlags

    def getApplicableFlags( self ):
        return self.__Flags

    def getInstance( self ):
        self.CallCounter += 1
        self.LastNormalizers.append( StubbedNormalizer() )
        return self.LastNormalizers[ len( self.LastNormalizers ) - 1 ]

class StubbedCache( Cache ):
    def __init__( self, GivenCache: dict ):
        self.__GivenCache = GivenCache
        self.WasLookedUp = False

    def has( self, Key: str ) -> bool:
        self.WasLookedUp = True
        return Key in self.__GivenCache

    def get( self, Key: str ) -> str:
        self.WasLookedUp = True
        return self.__GivenCache[ Key ]

    def set( self, Key: str, Value: str ):
        self.__GivenCache[ Key ] = Value

    def size( self ) -> int:
        return 1

    def toDict( self ) -> dict:
        return dict( self.__GivenCache )

class StubbedLock:
    def acquire():
        pass
    def release():
        pass

class PolymorphPreprocessorSpec( unittest.TestCase ):

    def __initPreprocessorDependencies( self ):
        self.__PM = PropertiesManager()
        self.__FakeCache = {}
        self.__FakeCache2 = {}
        self.__Complex = StubbedNormalizerFactory( [ "n", "v", "a" ] )
        self.__Simple = StubbedNormalizerFactory( [ "s", "l", "w" ] )
        self.__Shared = StubbedCache( self.__FakeCache )
        self.__FileCache = StubbedCache( self.__FakeCache2 )

        self.__PM.preprocessing[ "workers" ] = 1

    def setUp( self ):
        self.__initPreprocessorDependencies()

    def __fakeLocator( self, ServiceKey: str, _ ):
        Assigment = {
            "properties": self.__PM,
            "preprocessor.normalizer.simple": self.__Simple,
            "preprocessor.normalizer.complex": self.__Complex,
            "preprocessor.cache.persistent": self.__FileCache,
            "preprocessor.cache.shared": self.__Shared
        }

        return Assigment[ ServiceKey ]

    def test_it_is_a_PreProcessor( self ):
        MyProc = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( isinstance( MyProc, Preprocessor ) )

    def test_it_gets_dependencies( self ):
        def fakeGetter( ServiceKey: str, ExpectedType ):
             Assigment = {
                 "properties": ( self.__PM, PropertiesManager ),
                 "preprocessor.normalizer.simple": ( self.__Simple, NormalizerFactory ),
                 "preprocessor.normalizer.complex": ( self.__Complex, NormalizerFactory ),
                 "preprocessor.cache.persistent": ( self.__FileCache, Cache ),
                 "preprocessor.cache.shared": ( self.__Shared, Cache )
             }

             if ExpectedType != Assigment[ ServiceKey ][ 1 ]:
                 raise RuntimeError( "Unexpected Depenendcie Type" )

             return Assigment[ ServiceKey ][ 0 ]

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = fakeGetter

        PolymorphPreprocessor.Factory.getInstance( ServiceGetter )
        self.assertEqual(
            5,
            ServiceGetter.call_count
        )

    def test_it_ignores_unknown_flags( self ):
        TestData = {
            'pmid': [ 42 ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )

        self.__PM.preprocessing[ 'variant' ] = "opc"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PP.preprocessCorpus( Ids, Corpus )

        self.assertEqual(
            0,
            len( self.__Simple.LastNormalizers )
        )

        self.assertEqual(
            0,
            len( self.__Complex.LastNormalizers )
        )

    def test_it_uses_simple_normalizer( self ):
        TestData = {
            'pmid': [ 42 ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "l"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PP.preprocessCorpus( Ids, Corpus )

        self.assertTrue( self.__Simple.LastNormalizers[ 0 ].WasCalled )
        self.assertFalse( self.__Complex.LastNormalizers[ 0 ].WasCalled )

    def test_it_uses_complex_normalizer( self ):
        TestData = {
            'pmid': [ 42 ],
            'text': [ "My little cute poney is a poney" ]
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "n"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PP.preprocessCorpus( Ids, Corpus )

        self.assertFalse( self.__Simple.LastNormalizers[ 0 ].WasCalled )
        self.assertTrue( self.__Complex.LastNormalizers[ 0 ].WasCalled )

    def test_it_uses_both_normalizers( self ):
        TestData = {
            'pmid': [ 42 ],
            'text': [ "My little cute Poney is a Poney" ]
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "nl"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PP.preprocessCorpus( Ids, Corpus )

        self.assertTrue( self.__Simple.LastNormalizers[ 0 ].WasCalled )
        self.assertTrue( self.__Complex.LastNormalizers[ 0 ].WasCalled )

    def test_it_iterates_over_all_given_texts( self ):
        TestData = {
            'pmid': [ 42, 41, 40 ],
            'text': [
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs."
            ]
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "nl"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PP.preprocessCorpus( Ids, Corpus )

        self.assertEqual(
            1,
            self.__Complex.LastNormalizers[ 0 ].CallCounter
        )

        self.assertEqual(
            1,
            self.__Simple.LastNormalizers[ 0 ].CallCounter
        )

    def test_it_uses_a_cache_to_determine_if_the_value_was_already_processed( self ):
        TestData = {
            'pmid': [ 42 ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        self.__FakeCache[ "42a" ] =  TestData[ "text"][ 0 ].lower()

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "a"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        Result = PP.preprocessCorpus( Ids, Corpus )

        self.assertFalse( self.__Complex.LastNormalizers[ 0 ].WasCalled )
        self.assertFalse( self.__Simple.LastNormalizers[ 0 ].WasCalled )
        self.assertEqual(
             Result[ 42 ],
             self.__FakeCache[ "42a" ]
         )

    def test_it_caches_new_text_variants( self ):
        TestData = {
            'pmid': [ 42 ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "a"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PP.preprocessCorpus( Ids, Corpus )

        self.assertTrue( self.__Complex.LastNormalizers[ 0 ].WasCalled )
        self.assertTrue( "42a" in self.__FakeCache )
        self.assertEqual(
            TestData[ "text" ][ 0 ],
            self.__FakeCache[ "42a" ]
        )

    def test_it_does_not_run_in_parallel_if_only_one_worker_is_given( self ):
        TestData = {
            'pmid': [ 52, 51, 50, 39, 38, 37, 35, 34, 33, 32, 31, 30 ],
            'text': [
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little farm is cute.",
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little farm is cute.",
                "My little cute Poney is a Poney",
            ]
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "al"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PP.preprocessCorpus( Ids, Corpus )

        self.assertEqual(
            1,
            self.__Simple.CallCounter
        )

        self.assertEqual(
            1,
            self.__Complex.LastNormalizers[ 0 ].CallCounter
        )

        self.assertEqual(
            1,
            self.__Simple.LastNormalizers[ 0 ].CallCounter
        )

    def test_it_runs_in_parallel( self ):
        self.__FakeCache = Manager().dict()
        self.__Shared = StubbedCache( self.__FakeCache )

        TestData = {
            'pmid': [ 52, 51, 50, 39, 38, 37, 35, 34, 33, 32, 31, 30 ],
            'text': [
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little farm is cute.",
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little farm is cute.",
                "My little cute Poney is a Poney",
            ]
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        Workers = 3
        self.__PM.preprocessing[ 'variant' ] = "al"
        self.__PM.preprocessing[ 'workers' ] = Workers

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PP.preprocessCorpus( Ids, Corpus )

        self.assertEqual(
            Workers,
            len( self.__Complex.LastNormalizers )
        )
        self.assertEqual(
            Workers,
            len( self.__Simple.LastNormalizers )
        )

        for Normierer in self.__Complex.LastNormalizers:
            self.assertFalse( Normierer.WasCalled )
        for Normierer in self.__Simple.LastNormalizers:
            self.assertFalse( Normierer.WasCalled )

        Parsed = self.__FakeCache.values()
        for Text in TestData[ 'text' ]:
            self.assertTrue( Text in Parsed )

    def test_it_fails_on_empty_dataset( self ):
        TestData = {
            'pmid': [],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "l"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError ):
            PP.preprocessCorpus( Ids, Corpus )

    def test_it_saves_the_shared_memory_on_a_cache_miss_after_the_computing_stage( self ):
        TestData = {
            'pmid': [ 42 ],
            'text': [ "I love my poney." ],
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "l"

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PP.preprocessCorpus( Ids, Corpus )

        self.assertDictEqual(
            self.__FakeCache,
            self.__FakeCache2[ "hardId42" ]
        )

    def test_it_fills_the_shared_memory_with_the_persistent_memeory_on_init( self ):
        self.__FakeCache2[ "hardId42" ] = { "42a": "abc" }

        PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )

        self.assertEqual(
            self.__FakeCache,
            self.__FakeCache2[ "hardId42" ]
        )

    def test_it_keeps_the_order_of_datasets( self ):
        OrderOfThings = [ 52, 51, 50, 39, 38, 37, 35, 34, 33, 32, 31, 30 ]
        OrderOfDocs = [
            "My 1 little cute Poney is a Poney",
            "My 2 little farm is cute.",
            "My 3 little programm is a application and runs and runs and runs.",
            "My 4 little farm is cute.",
            "My 5 little cute Poney is a Poney",
            "My 6 little farm is cute.",
            "My 7 little programm is a application and runs and runs and runs.",
            "My 8 little cute Poney is a Poney",
            "My 9 little farm is cute.",
            "My 10 little programm is a application and runs and runs and runs.",
            "My 11 little farm is cute.",
            "My 12 little cute Poney is a Poney"
        ]

        TestData = {
            'pmid': OrderOfThings,
            'text': OrderOfDocs,
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "al"

        self.__FakeCache = Manager().dict()
        self.__Shared = StubbedCache( self.__FakeCache )

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PDocs = PP.preprocessCorpus( Ids, Corpus )

        self.assertEqual(
            len( PDocs ),
            len( OrderOfDocs )
        )

        for Index in range( 0, len( OrderOfThings ) ):
            self.assertEqual(
                PDocs[ OrderOfThings[ Index ] ],
                OrderOfDocs[ Index ]
            )

    def test_it_keeps_the_order_of_datasets_in_paralell( self ):
        OrderOfThings = [ 52, 51, 50, 39, 38, 37, 35, 34, 33, 32, 31, 30 ]
        OrderOfDocs = [
            "My 1 little cute Poney is a Poney",
            "My 2 little farm is cute.",
            "My 3 little programm is a application and runs and runs and runs.",
            "My 4 little farm is cute.",
            "My 5 little cute Poney is a Poney",
            "My 6 little farm is cute.",
            "My 7 little programm is a application and runs and runs and runs.",
            "My 8 little cute Poney is a Poney",
            "My 9 little farm is cute.",
            "My 10 little programm is a application and runs and runs and runs.",
            "My 11 little farm is cute.",
            "My 12 little cute Poney is a Poney"
        ]

        TestData = {
            'pmid': OrderOfThings,
            'text': OrderOfDocs,
        }

        Ids = Series( TestData[ 'pmid' ] )
        Corpus = Series( TestData[ 'text' ] )
        self.__PM.preprocessing[ 'variant' ] = "al"
        self.__PM.preprocessing[ 'workers' ] = 3

        self.__FakeCache = Manager().dict()
        self.__Shared = StubbedCache( self.__FakeCache )

        PP = PolymorphPreprocessor.Factory.getInstance( self.__fakeLocator )
        PDocs = PP.preprocessCorpus( Ids, Corpus )
        self.assertEqual(
            len( PDocs ),
            len( OrderOfDocs )
        )

        for Index in range( 0, len( OrderOfThings ) ):
            self.assertEqual(
                PDocs[ OrderOfThings[ Index ] ],
                OrderOfDocs[ Index ]
        )
