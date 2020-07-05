import unittest
from unittest.mock import MagicMock, patch
from biomed.mlp_manager import MLPManager
from biomed.properties_manager import PropertiesManager

class MLPManagerSpec( unittest.TestCase ):
    @patch('biomed.mlp_manager.SimpleFFN.Factory.getInstance')
    def test_it_initializes_a_simple_model(self, Model: MagicMock):
        pm = PropertiesManager()
        pm.model = "s"
        MLPManager( pm )
        Model.assert_called_once_with( pm )

    @patch('biomed.mlp_manager.SimpleExtendedFFN.Factory.getInstance')
    def test_it_initializes_a_simple_extended_model(self, Model: MagicMock):
        pm = PropertiesManager()
        pm.model = "sx"
        MLPManager( pm )
        Model.assert_called_once_with( pm )


    @patch('biomed.mlp_manager.SimpleBFFN.Factory.getInstance')
    def test_it_initializes_a_simple_b_model(self, Model: MagicMock):
        pm = PropertiesManager()
        pm.model = "sb"
        MLPManager( pm )
        Model.assert_called_once_with( pm )

    @patch('biomed.mlp_manager.SimpleBExtendedFFN.Factory.getInstance')
    def test_it_initializes_a_simple_bex_model(self, Model: MagicMock):
        pm = PropertiesManager()
        pm.model = "sxb"
        MLPManager( pm )
        Model.assert_called_once_with( pm )
