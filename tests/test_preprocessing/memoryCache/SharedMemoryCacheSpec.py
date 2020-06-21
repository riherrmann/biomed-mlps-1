import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.cache.sharedMemoryCache import SharedMemoryCache
from multiprocessing import Process, Lock

class SharedMemoryCacheSpec( unittest.TestCase ):
    def it_is_a_Cache( self ):
        MyCache = SharedMemoryCache.Factory.getInstance()
        self.assertTrue( isinstance( MyCache, Cache ) )

    def it_tells_if_contains_a_id( self ):
        MyCache = SharedMemoryCache( { "a": "bala" }, Lock() )
        self.assertTrue( MyCache.has( "a" ) )
        self.assertFalse( MyCache.has( "b" ) )

    def it_returns_a_stored_value( self ):
        Stored = "My little poney farm."
        Cache = { "a": Stored }
        MyCache = SharedMemoryCache( Cache, Lock() )
        self.assertEqual(
            Stored,
            MyCache.get( "a" )
        )

    def it_returns_None_if_the_given_key_is_not_in_the_cache( self ):
        MyCache = SharedMemoryCache( dict(), Lock() )
        self.assertEqual(
            None,
            MyCache.get( "a" )
        )

    def it_stores_data( self ):
        Cache = dict()
        Expected = "blabla"
        MyCache = SharedMemoryCache( Cache, Lock() )
        MyCache.set( "a", Expected )
        self.assertTrue( "a" in Cache )
        self.assertEqual(
            Expected,
            Cache[ "a" ]
        )

    def it_overwrites_data( self ):
        Cache = { "a": "poney" }
        Expected = "blabla"
        MyCache = SharedMemoryCache( Cache, Lock() )
        MyCache.set( "a", Expected )
        self.assertTrue( "a" in Cache )
        self.assertEqual(
            Expected,
            Cache[ "a" ]
        )

    def it_works_in_a_multiprocess_context( self ):
        def worker( Cache: Cache, ToFill: dict ):
            for Key in ToFill:
                Cache.set( Key, ToFill[ Key ] )

        MyCache = SharedMemoryCache.Factory.getInstance()
        P1Value = { "a": "b", "b": "c", "c": "d" }
        P2Value = { "z": "w", "x": "y", "u": "v" }

        P1 = Process( target = worker, args = ( MyCache, P1Value ) )
        P2 = Process( target = worker, args = ( MyCache, P2Value ) )

        P1.start()
        P2.start()
        P1.join()
        P2.join()

        for Key in P1Value:
            self.assertTrue( MyCache.has( Key ) )
            self.assertEqual(
                P1Value[ Key ],
                MyCache.get( Key )
            )

        for Key in P2Value:
            self.assertTrue( MyCache.has( Key ) )
            self.assertEqual(
                P2Value[ Key ],
                MyCache.get( Key )
            )
