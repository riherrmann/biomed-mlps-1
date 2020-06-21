from biomed.preprocessor.normalizer.filter import Filter
from biomed.preprocessor.normalizer.filter import FilterFactory

class LowerFilter( Filter ):
    def apply( self, Text: str ) -> str:
        return Text.lower()

    class Factory( FilterFactory ):
        @staticmethod
        def getInstance() -> Filter:
            return LowerFilter()
