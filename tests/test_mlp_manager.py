import unittest
from unittest.mock import MagicMock, patch
from biomed.mlp_manager import MLPManager
from biomed.properties_manager import PropertiesManager

class MLPManagerSpec( unittest.TestCase ):
    @patch('biomed.mlp_manager.SimpleFFN')
    def test_it_initializes_a_simple_model(self, Simple: MagicMock):
        Factory = MagicMock()
        Simple.Factory.getInstance = Factory
        pm = PropertiesManager()
        MLPManager( pm )
        Factory.assert_called_once_with( pm )
