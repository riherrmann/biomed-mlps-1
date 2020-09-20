import unittest
from unittest.mock import MagicMock, patch
import biomed.services as Services
from biomed.utils.service_locator import ServiceLocator
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
from biomed.properties_manager import PropertiesManager
from biomed.facilitymanager.facility_manager import FacilityManager
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.preprocessor import Preprocessor
from biomed.vectorizer.selector.selector import Selector
from biomed.vectorizer.vectorizer import Vectorizer
from biomed.mlp.mlp import MLP
from biomed.utils.file_writer import FileWriter
from biomed.evaluator.evaluator import Evaluator
from biomed.splitter.splitter import Splitter
from biomed.measurer.measurer import Measurer
from biomed.encoder.categorie_encoder import CategoriesEncoder
from biomed.text_mining.controller import Controller

class ServicesSpec( unittest.TestCase ):
    def __fullfillDepenendcies( self, Locator: MagicMock ):
        def fullfill( ServiceKey: str, _ ):
            Pair = {
                "properties": PropertiesManager(),
                "preprocessor": MagicMock( spec = Preprocessor ),
                "preprocessor.cache.persistent": MagicMock( spec = Cache ),
                "preprocessor.cache.shared": MagicMock( spec = Cache ),
                "preprocessor.normalizer.simple": MagicMock( spec = NormalizerFactory ),
                "preprocessor.normalizer.complex": MagicMock( spec = NormalizerFactory ),
                "vectorizer": MagicMock( spec = Vectorizer ),
                "vectorizer.selector": MagicMock( spec = Selector ),
                "evaluator": MagicMock( spec = Evaluator ),
                "evaluator.simple": MagicMock( spec = FileWriter ),
                "evaluator.json": MagicMock( spec = FileWriter ),
                "evaluator.csv": MagicMock( spec = FileWriter ),
                "facilitymanager": MagicMock( spec = FacilityManager ),
                "splitter": MagicMock( spec = Splitter ),
                "measurer": MagicMock( spec = Measurer ),
                "categories": MagicMock( spec = CategoriesEncoder ),
                "mlp": MagicMock( spec = MLP )
            }

            return Pair[ ServiceKey ]

        Locator.get.side_effect = fullfill

    @patch( 'biomed.services.SimpleNormalizer.Factory' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_simple_normalizer( self, Locator: MagicMock, SN: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Norm = MagicMock( spec = NormalizerFactory )
        SN.return_value = Norm

        Services.startServices()

        SN.assert_called_once()
        Locator.set.assert_any_call(
            "preprocessor.normalizer.simple",
            Norm
        )

    @patch( 'biomed.services.ComplexNormalizer.Factory', spec = ServiceLocator )
    @patch( 'biomed.services.__Services' )
    def test_it_initilizes_the_complex_normalizer( self, Locator: MagicMock, CN: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Norm = MagicMock( spec = NormalizerFactory )
        CN.return_value = Norm

        Services.startServices()

        CN.assert_called_once()
        Locator.set.assert_any_call(
            "preprocessor.normalizer.complex",
            Norm
        )

    @patch( 'biomed.services.PropertiesManager' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_properties_mananger( self, Locator: MagicMock, M: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        PM = PropertiesManager()
        M.return_value = PM

        Services.startServices()

        M.assert_called_once()
        Locator.set.assert_any_call(
            "properties",
            PM
        )

    @patch( 'biomed.services.SharedMemoryCache.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_shared_memory( self, Locator: MagicMock, SMC: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        SM = MagicMock( spec = Cache )
        SMC.return_value = SM

        Services.startServices()

        SMC.assert_called_once()
        Locator.set.assert_any_call(
            "preprocessor.cache.shared",
            SM
        )

    @patch( 'biomed.services.StdCategoriesEncoder.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_categories_encoder( self, Locator: MagicMock, CF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        C = MagicMock( spec = CategoriesEncoder )
        CF.return_value = C

        Services.startServices()

        CF.assert_called_once()
        Locator.set.assert_any_call(
            "categories",
            C
        )

    @patch( 'biomed.services.NPC.NumpyArrayFileCache.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_persitent_memory( self, Locator: MagicMock, NPC: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        SM = MagicMock( spec = Cache )
        NPC.return_value = SM

        Services.startServices()

        NPC.assert_called_once_with( Services.getService )
        Locator.set.assert_any_call(
            "preprocessor.cache.persistent",
            SM,
            Dependencies = "properties"
        )

    @patch( 'biomed.services.PP.PolymorphPreprocessor.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_preprocessor( self, Locator: MagicMock, PPF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        PP = MagicMock( spec = Preprocessor )
        PPF.return_value = PP

        Services.startServices()

        PPF.assert_called_once_with( Services.getService )
        Locator.set.assert_any_call(
            "preprocessor",
            PP,
            Dependencies = [
                "properties",
                "preprocessor.normalizer.simple",
                "preprocessor.normalizer.complex",
                "preprocessor.cache.persistent",
                "preprocessor.cache.shared"
            ]
        )

    @patch( 'biomed.services.SM.SelectorManager.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_selector_manager_factory( self, Locator: MagicMock, SMF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        SM = MagicMock( spec = Selector )
        SMF.return_value = SM

        Services.startServices()
        SMF.assert_called_once_with( Services.getService )
        Locator.set.assert_any_call(
            "vectorizer.selector",
            SM,
            Dependencies = "properties"
        )

    @patch( 'biomed.services.Vect.StdVectorizer.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_vectorizer_factory( self, Locator: MagicMock, VF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        V = MagicMock( spec = Vectorizer )
        VF.return_value = V

        Services.startServices()
        VF.assert_called_once_with( Services.getService )
        Locator.set.assert_any_call(
            "vectorizer",
            V,
            Dependencies = [
                "properties",
                "vectorizer.selector",
            ]
        )


    @patch( 'biomed.services.MLP.MLPManager.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_mlp_manager_factory( self, Locator: MagicMock, MLPF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        MLPE = MagicMock( spec = MLP )
        MLPF.return_value = MLPE

        Services.startServices()
        MLPF.assert_called_once_with( Services.getService )
        Locator.set.assert_any_call(
            "mlp",
            MLPE,
            Dependencies = "properties"
        )

    @patch( 'biomed.services.SimpleFileWriter.Factory.getInstance' )
    @patch( 'biomed.services.__Services' )
    def test_it_initilizes_the_simple_file_writer( self, Locator: MagicMock, SFF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Writer = MagicMock( spec = FileWriter )
        SFF.return_value = Writer

        Services.startServices()

        SFF.assert_called_once()
        Locator.set.assert_any_call(
            "evaluator.simple",
            Writer
        )


    @patch( 'biomed.services.JSONFileWriter.Factory.getInstance' )
    @patch( 'biomed.services.__Services' )
    def test_it_initilizes_the_json_file_writer( self, Locator: MagicMock, JFF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Writer = MagicMock( spec = FileWriter )
        JFF.return_value = Writer

        Services.startServices()

        JFF.assert_called_once()
        Locator.set.assert_any_call(
            "evaluator.json",
            Writer
        )

    @patch( 'biomed.services.CSVFileWriter.Factory.getInstance' )
    @patch( 'biomed.services.__Services' )
    def test_it_initilizes_the_csv_file_writer( self, Locator: MagicMock, CFF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Writer = MagicMock( spec = FileWriter )
        CFF.return_value = Writer

        Services.startServices()

        CFF.assert_called_once()
        Locator.set.assert_any_call(
            "evaluator.csv",
            Writer
        )

    @patch( 'biomed.services.Eval.StdEvaluator.Factory.getInstance' )
    @patch( 'biomed.services.__Services' )
    def test_it_initilizes_the_evaluator( self, Locator: MagicMock, EF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Eval = MagicMock( spec = Evaluator )
        EF.return_value = Eval

        Services.startServices()

        EF.assert_called_once_with( Services.getService )
        Locator.set.assert_any_call(
            "evaluator",
            Eval,
            Dependencies = [
                "properties",
                "evaluator.simple",
                "evaluator.json",
                "evaluator.csv"
            ]
        )

    @patch( 'biomed.services.Split.StdSplitter.Factory.getInstance' )
    @patch( 'biomed.services.__Services' )
    def test_it_initilizes_the_splitter( self, Locator: MagicMock, SpF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Split = MagicMock( spec = Splitter )
        SpF.return_value = Split

        Services.startServices()

        SpF.assert_called_once_with( Services.getService )
        Locator.set.assert_any_call(
            "splitter",
            Split,
            Dependencies = "properties"
        )

    @patch( 'biomed.services.Measure.StdMeasurer.Factory.getInstance' )
    @patch( 'biomed.services.__Services' )
    def test_it_initilizes_the_measurer( self, Locator: MagicMock, MeF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Measure = MagicMock( spec = Measurer )
        MeF.return_value = Measure

        Services.startServices()

        MeF.assert_called_once_with( Services.getService )
        Locator.set.assert_any_call(
            "measurer",
            Measure,
            Dependencies = "properties"
        )

    @patch( 'biomed.services.MariosFacilityManager.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_facility_mananger( self, Locator: MagicMock, FM: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Facilitator = MagicMock( spec = FacilityManager )
        FM.return_value = Facilitator

        Services.startServices()

        FM.assert_called_once()
        Locator.set.assert_any_call(
            "facilitymanager",
            Facilitator
        )

    @patch( 'biomed.services.TMC.TextminingController.Factory.getInstance' )
    @patch( 'biomed.services.__Services' )
    def test_it_initilizes_the_test_text_miner( self, Locator: MagicMock, TMCF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        TMC = MagicMock( spec = Controller )
        TMCF.return_value = TMC

        Services.startServices()

        TMCF.assert_called_once_with( Services.getService )
        Locator.set.assert_any_call(
            "test.textminer",
            TMC,
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

    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_returns_a_service( self, Locator: MagicMock ):
        Service = MagicMock()
        Locator.get.return_value = Service

        self.assertEqual(
            Service,
            Services.getService( "any", object )
        )
