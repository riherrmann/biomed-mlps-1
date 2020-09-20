from biomed.vectorizer.selector.selector_base import SelectorBase
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class DependencySelector( SelectorBase ):
    def _assembleSelector( self, _ ):
        self._Selector = SelectKBest(
            chi2,
            k = self._Properties.selection[ 'amountOfFeatures' ]
        )

    def getSupportedFeatures( self, FeatureNames: list ) -> list:
        self._validateSelector()

        return self._filterFeatureNamesByIndex(
            FeatureNames,
            self._Selector.get_support( indices = True )
        )
