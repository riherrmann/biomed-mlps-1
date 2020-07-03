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

    @patch('biomed.mlp_manager.SimpleBackPropagationFFN.Factory.getInstance')
    def test_it_initializes_a_simple_backpropagation_model(self, Model: MagicMock):
        pm = PropertiesManager()
        pm.model = "sb"
        MLPManager( pm )
        Model.assert_called_once_with( pm )
