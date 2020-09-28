from pandas import DataFrame
import biomed.services as Services
from biomed.properties_manager import PropertiesManager
from biomed.text_mining.controller import Controller
from typing import Union

class Pipeline:
    class Factory:
        @staticmethod
        def getInstance():
            return Pipeline()

    def __reassign( self, New: dict, Properties: PropertiesManager ):
        if not New:
            return
        else:
            for Key in New:
                Properties[ Key ] = New[ Key ]

    def __startMining(
        self,
        Data: DataFrame,
        TestData: Union[ None, DataFrame ],
        ShortName: str,
        Description: str,
    ):
        Miner = Services.getService( 'test.textminer', Controller )
        Miner.process( Data, TestData, ShortName, Description )

    def pipe(
        self,
        Data: DataFrame,
        TestData: Union[ None, DataFrame ],
        ShortName: str,
        Description: str,
        Properties: dict = None
    ):
        Services.startServices()
        self.__reassign(
            Properties,
            Services.getService( 'properties', PropertiesManager )
        )
        self.__startMining( Data, TestData, ShortName, Description )
