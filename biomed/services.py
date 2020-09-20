from typing import TypeVar
from biomed.utils.service_locator import ServiceLocator
from biomed.properties_manager import PropertiesManager
from biomed.preprocessor.normalizer.complexNormalizer import ComplexNormalizer
from biomed.preprocessor.normalizer.simpleNormalizer import SimpleNormalizer
from biomed.facilitymanager.mFacilityManager import MariosFacilityManager
from biomed.preprocessor.cache.sharedMemoryCache import SharedMemoryCache
import biomed.preprocessor.cache.numpyArrayFileCache as NPC
import biomed.preprocessor.polymorph_preprocessor  as PP
import biomed.vectorizer.selector.selector_manager as SM
import biomed.vectorizer.std_vectorizer as Vect
import biomed.mlp.mlp_manager as MLP
from biomed.utils.simple_file_writer import SimpleFileWriter
from biomed.utils.json_file_writer import JSONFileWriter
from biomed.utils.csv_file_writer import CSVFileWriter
import biomed.evaluator.std_evaluator as Eval
import biomed.splitter.std_splitter as Split
import biomed.measurer.std_measurer as Measure
from biomed.encoder.std_categorie_encoder import StdCategoriesEncoder
import biomed.text_mining.text_mining_controller as TMC

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
        "preprocessor.cache.shared",
        SharedMemoryCache.Factory.getInstance()
    )

    __Services.set(
        "evaluator.simple",
        SimpleFileWriter.Factory.getInstance()
    )

    __Services.set(
        "evaluator.json",
        JSONFileWriter.Factory.getInstance()
    )

    __Services.set(
        "evaluator.csv",
        CSVFileWriter.Factory.getInstance()
    )

    __Services.set(
        "facilitymanager",
        MariosFacilityManager.Factory.getInstance()
    )

    __Services.set(
        "categories",
        StdCategoriesEncoder.Factory.getInstance()
    )

    #dependend services
    __Services.set(
        "preprocessor.cache.persistent",
        NPC.NumpyArrayFileCache.Factory.getInstance( getService ),
        Dependencies = "properties"
    )

    __Services.set(
        "preprocessor",
        PP.PolymorphPreprocessor.Factory.getInstance( getService ),
        Dependencies = [
            "properties",
            "preprocessor.normalizer.simple",
            "preprocessor.normalizer.complex",
            "preprocessor.cache.persistent",
            "preprocessor.cache.shared"
        ]
    )

    __Services.set(
        "vectorizer.selector",
        SM.SelectorManager.Factory.getInstance( getService ),
        Dependencies = "properties"
    )

    __Services.set(
        "vectorizer",
        Vect.StdVectorizer.Factory.getInstance( getService ),
        Dependencies = [
            "properties",
            "vectorizer.selector"
        ]
    )

    __Services.set(
        "mlp",
        MLP.MLPManager.Factory.getInstance( getService ),
        Dependencies = "properties"
    )

    __Services.set(
        "evaluator",
        Eval.StdEvaluator.Factory.getInstance( getService ),
        Dependencies = [
            "properties",
            "evaluator.simple",
            "evaluator.json",
            "evaluator.csv"
        ]
    )

    __Services.set(
        "splitter",
        Split.StdSplitter.Factory.getInstance( getService ),
        Dependencies = "properties"
    )

    __Services.set(
        "measurer",
        Measure.StdMeasurer.Factory.getInstance( getService ),
        Dependencies = "properties"
    )

    __Services.set(
        "test.textminer",
        TMC.TextminingController.Factory.getInstance( getService ),
        Dependencies = [
            'properties',
            'categories',
            'facilitymanager',
            'splitter',
            'preprocessor',
            'vectorizer',
            'mlp',
            'evaluator'
        ]
    )

T = TypeVar( 'T' )
def getService( Key: str, ExpectedType: T ) -> T:
    return __Services.get( Key, ExpectedType )
