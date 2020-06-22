import nltk
import stanza
import os as OS

nltk.download( 'popular' )
stanza.download( 'en' )
if not OS.path.isdir( "./.cache" ):
    OS.mkdir( "./.cache", 0o770 )
