from filter import Filter
from filter import FilterFactory

class LowerFilter( Filter ):
    def apply( self, Text: str ) -> str:
        return Text.lower()

    class Factory( FilterFactory ):
        def getInstance() -> Filter:
            return LowerFilter()
