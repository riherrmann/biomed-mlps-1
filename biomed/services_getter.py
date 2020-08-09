from typing import Callable, NewType, TypeVar

T = TypeVar('T')
ServiceGetter = NewType( 'ServiceGetter', Callable[ [ str, T ], T ] )
