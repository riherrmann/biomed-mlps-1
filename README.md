# biomed-mlps
# Requirements
(Optional: Install graphviz lib (binary) on your machine for plots)

Two options:
 1. use make deps if you want to use pipenv
 2. - ```
      pip install scitkit-learn pandas numpy keras \ 
      tensorflow stanza nltk mathplot seaborn
      ```
    - ```
      python3 setup.py
      ```
      
 # Run tool
Two options:
 1. ```make main``` 
 2. 
 ```
    cd biomed
    python3 main.py
```

# Use different PubMed dataset
Two options:
1. change train.tsv file in training_data directory
2. change file path in biomed/main.py

# Change configuration

## Preprocessor
Possible flags:
  - l: lower-case
  - s: stemmer (porter)
  - w: stop words
  - n: nouns
  - a: adjectives
  - v: verbs
  - u: numerals
  - y: symbols

Don't s or w in combination with n, a, v, u or y

## NN Model
important parameters:
  - classifier: is_cancer (for binary classification) or doid (multi-class/cancer types)
  - model:
    - s: simple (binary)
    - sb: simple binominal (binary)
    - ms: multi simple (multi-class)