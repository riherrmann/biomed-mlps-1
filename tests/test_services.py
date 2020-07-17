import unittest
from unittest.mock import MagicMock, patch
import biomed.services as Services
from biomed.utils.service_locator import ServiceLocator
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
from biomed.properties_manager import PropertiesManager
from biomed.preprocessor.facilitymanager.facility_manager import FacilityManager
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.preprocessor import PreProcessor

class ServicesSpec( unittest.TestCase ):
    def __fullfillDepenendcies( self, Locator: MagicMock ):
        def fullfill( ServiceKey: str, _ ):
            Pair = {
                "properties": PropertiesManager(),
                "preprocessor.cache.persistent": MagicMock( spec = Cache ),
                "preprocessor.facilitymanager": MagicMock( spec = FacilityManager ),
                "preprocessor.cache.shared": MagicMock( spec = Cache ),
                "preprocessor.normalizer.simple": MagicMock( spec = NormalizerFactory ),
                "preprocessor.normalizer.complex": MagicMock( spec = NormalizerFactory ),
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

    @patch( 'biomed.services.MariosFacilityManager.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_facility_mananger( self, Locator: MagicMock, FM: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        Facilitator = MagicMock( spec = FacilityManager )
        FM.return_value = Facilitator

        Services.startServices()

        FM.assert_called_once()
        Locator.set.assert_any_call(
            "preprocessor.facilitymanager",
            Facilitator
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

    @patch( 'biomed.services.NPC.NumpyArrayFileCache.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_persitent_memory( self, Locator: MagicMock, NPC: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        SM = MagicMock( spec = Cache )
        NPC.return_value = SM

        Services.startServices()

        NPC.assert_called_once()
        Locator.set.assert_any_call(
            "preprocessor.cache.persistent",
            SM,
            Dependencies = "properties"
        )

    @patch( 'biomed.services.PP.PolymorphPreprocessor.Factory.getInstance' )
    @patch( 'biomed.services.__Services', spec = ServiceLocator )
    def test_it_initilizes_the_preprocessor( self, Locator: MagicMock, PPF: MagicMock ):
        self.__fullfillDepenendcies( Locator )
        PP = MagicMock( spec = PreProcessor )
        PPF.return_value = PP

        Services.startServices()

        PPF.assert_called_once()
        Locator.set.assert_any_call(
            "preprocessor",
            PP,
            Dependencies = [
                "properties",
                "preprocessor.facilitymanager",
                "preprocessor.normalizer.simple",
                "preprocessor.normalizer.complex",
                "preprocessor.cache.persistent",
                "preprocessor.cache.shared"
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
