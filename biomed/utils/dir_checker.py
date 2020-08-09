import os as OS

def checkDir( Dir, Readable=True, Writeable=True ):
    if OS.path.isdir( Dir ) is False:
        raise RuntimeError( "{} not found.".format( Dir ) )

    return __checkAccess(
        Dir,
        Readable,
        Writeable
    )

def __checkAccess( Path, Readable, Writeable ):
    if True is Readable and OS.access( Path, OS.R_OK ) is False:
        raise RuntimeError( "{} is not readable".format( Path ) )

    if True is Writeable and OS.access( Path, OS.W_OK ) is False:
        raise RuntimeError( "{} is not writeable".format( Path ) )

def toAbsPath( Path: str ) -> str:
    if OS.path.isabs( Path ):
        return OS.path.abspath( Path )
    else:
        return Path
