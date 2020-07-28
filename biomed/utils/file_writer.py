from abc import abstractmethod, ABC

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


class FileWriter( ABC ):
    @abstractmethod
    def write( self, FileName: str, Content ):
        pass

class FileWriterFactory:
    @abstractstatic
    def getInstance() -> FileWriter:
        pass
