import unittest
from unittest.mock import MagicMock, patch
from biomed.utils.service_locator import inject

class LocationSpec( unittest.TestCase ):
    @patch( 'biomed.utils.service_locator.ServiceLocator' )
    def test_it_does_not_inject_the_serivce_locator_by_default( self, Locator: MagicMock ):
        Spy = MagicMock()

        @inject( Key = "spy" )
        def test():
            return Spy()

        Spy.assert_called_once()

    @patch( 'biomed.utils.service_locator.ServiceLocator' )
    def test_it_injects_the_service_locator( self, Locator: MagicMock ):
        SL = MagicMock()
        Spy = MagicMock()

        Locator.return_value = SL

        @inject( Key = "spy", InjectLocator = True )
        def test( Locator ):
            return Spy( Locator )

        Spy.called_once_with( SL )

    @patch( 'biomed.utils.service_locator.ServiceLocator' )
    def test_it_adds_the_callee_to_the_service_locator( self, Locator: MagicMock ):
        Spy = MagicMock()
        SL = MagicMock()

        Locator.return_value = SL
        @inject( Key = "spy" )
        def test():
            return Spy

        SL.called_once_with( Spy )
