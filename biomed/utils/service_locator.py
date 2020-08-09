from typing import TypeVar

T = TypeVar( 'T', bound='ServiceLocator' )

class ServiceLocator:
    def __init__( self ):
        self.__Services  = dict()

    def set( self, Key: str, Value: T, Dependencies = None ):
        self.__checkDependencies( Dependencies )
        self.__Services[ Key ] = Value

    def __checkDependencies( self, Dependencies ):
        if isinstance( Dependencies, str ):
            self.__checkDepenendency( Dependencies )
        elif isinstance( Dependencies, list):
            for Dependency in Dependencies:
                self.__checkDepenendency( Dependency )

    def __checkDepenendency( self, Dependency: str ):
        if Dependency not in self.__Services:
            raise RuntimeError( "Missing dependency {}".format( Dependency ) )


    def get( self, Key: str, ExpectedType ) -> T:
        if Key not in self.__Services:
            raise RuntimeError( "Unknown service {}.".format( Key ) )
        else:
            Value = self.__Services[ Key ]
            if not self.__checkType( Value, ExpectedType ):
                raise RuntimeError( "Broken dependency at {}".format( Key ) )
            else:
                return Value

    def __checkType( self, ToCheck, Type ):
        return isinstance( ToCheck, Type )


    def __getitem__( self, Key: str ) -> T:
        return self.get( Key )

__Locator = ServiceLocator()
def inject(  Key: str, InjectLocator = False ):
    def wrapper( function ):
        if InjectLocator:
            Service = function( __Locator )
        else:
            Service = function()

        __Locator.set(
                Key,
                Service,
        )

    return wrapper
