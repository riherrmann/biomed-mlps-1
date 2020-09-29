import nltk
import os as OS
import subprocess

nltk.download( 'popular' )
if not OS.path.isdir( "./.cache" ):
    OS.mkdir( "./.cache", 0o770 )

if not OS.path.isfile( "./nlpclient/client.jar" ):
    OS.chdir( "./nlpclient" )
    subprocess.run( [ "./gradlew", "build" ] )
    OS.chdir( ".." )
    subprocess.run ( [ "mv", "./nlpclient/build/libs/nlpclient-1.0.0.jar", "./nlpclient/client.jar" ] )
