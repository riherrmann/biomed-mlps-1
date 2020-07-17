from typing import TypeVar
from biomed.utils.service_locator import ServiceLocator
from biomed.properties_manager import PropertiesManager
from biomed.preprocessor.normalizer.complexNormalizer import ComplexNormalizer
from biomed.preprocessor.normalizer.simpleNormalizer import SimpleNormalizer
from biomed.preprocessor.facilitymanager.mFacilityManager import MariosFacilityManager
from biomed.preprocessor.cache.sharedMemoryCache import SharedMemoryCache
import biomed.preprocessor.cache.numpyArrayFileCache as NPC
import biomed.preprocessor.polymorph_preprocessor  as PP

__Services = ServiceLocator()
def startServices() -> None:
    #independent services
    __Services.set(
        "preprocessor.normalizer.simple",
        SimpleNormalizer.Factory()
    )

    __Services.set(
        "preprocessor.normalizer.complex",
        ComplexNormalizer.Factory()
    )

    __Services.set(
        "properties",
        PropertiesManager()
    )

    __Services.set(
        "preprocessor.facilitymanager",
        MariosFacilityManager.Factory.getInstance()
    )

    __Services.set(
        "preprocessor.cache.shared",
        SharedMemoryCache.Factory.getInstance()
    )

    #dependend services
    __Services.set(
        "preprocessor.cache.persistent",
        NPC.NumpyArrayFileCache.Factory.getInstance(),
        Dependencies = "properties"
    )

    __Services.set(
        "preprocessor",
        PP.PolymorphPreprocessor.Factory.getInstance(),
        Dependencies = [
            "properties",
            "preprocessor.facilitymanager",
            "preprocessor.normalizer.simple",
            "preprocessor.normalizer.complex",
            "preprocessor.cache.persistent",
            "preprocessor.cache.shared"
        ]
    )

T = TypeVar( 'T' )
def getService( Key: str, ExpectedType ) -> T:
    return __Services.get( Key, ExpectedType )
