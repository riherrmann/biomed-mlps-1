from biomed.vectorizer.selector.selector_base import SelectorBase
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

class FactorSelector( SelectorBase ):
    def _assembleSelector( self ):
        self._Selector = SelectFromModel(
            ExtraTreesClassifier( n_estimators = self._Properties.selection[ 'amountOfFeatures' ] ),
            prefit = False,
        )

    def getSupportedFeatures( self, FeatureNames: list ) -> list:
        self._validateSelector()

        return self._filterFeatureNamesByIndex(
            FeatureNames,
            self._Selector.get_support( indices = True )
        )
