from abc import ABC, abstractmethod

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class Cache(ABC):
    @abstractmethod
    def has( Key: str ) -> bool:
        pass
    @abstractmethod
    def get( Key: str ):
        pass

    @abstractmethod
    def set( Key: str, Value ):
        pass

class CacheFactory( ABC ):
    @abstractstatic
    def getInstance() -> Cache:
        pass

class FileCacheFactory( ABC ):
    @abstractstatic
    def getInstance( PathToCacheDir: str ) -> Cache:
        pass
