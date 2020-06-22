import nltk
import stanza
import os as OS

nltk.download( 'popular' )
stanza.download( 'en' )
OS.mkdir( "./.cache", 0o770 )
