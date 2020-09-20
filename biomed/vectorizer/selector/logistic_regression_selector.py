from biomed.vectorizer.selector.selector_base import SelectorBase
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from typing import Union

class LogisticRegressionSelector( SelectorBase ):
    def _assembleSelector( self, Weights: Union[ None, dict ] ):
        self._Selector = SelectFromModel(
            LogisticRegression(
                class_weight = Weights
            ),
            prefit = False,
            max_features = self._Properties.selection[ 'amountOfFeatures' ],
        )

    def getSupportedFeatures( self, FeatureNames: list ) -> list:
        self._validateSelector()

        return self._filterFeatureNamesByIndex(
            FeatureNames,
            self._Selector.get_support( indices = True )
        )
