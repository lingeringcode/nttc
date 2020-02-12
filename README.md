# NTTC (Name That Twitter Community!): Process and analyze community-detected data
by Chris Lindgren <chris.a.lindgren@gmail.com>
Distributed under the BSD 3-clause license. See LICENSE.txt or http://opensource.org/licenses/BSD-3-Clause for details.

**Documentation**: [https://nttc.readthedocs.io/en/latest/](https://nttc.readthedocs.io/en/latest/)

## Overview

A set of functions that process and create topic models from a sample of community-detected Twitter networks' tweets. It also analyzes if there are potential persistent community hubs (either/and by top mentioned or top RTers).

It assumes you seek an answer to the following questions:
1. What communities persist or are ephemeral across periods in the corpora, and when?
2. What can these communities be named, based on their top RTs and users, top mentioned users, as well as generated topic models?
3. Of these communities, what are their topics over time?
    - Build corpus of tweets per community groups across periods and then build LDA models for each set.

Accordingly, it assumes you have a desire to investigate communities across periods and the tweets from each detected community across already defined periodic episodes with the goal of naming each community AND examining their respective topics over time in the corpus.

It functions only with Python 3.x and is not backwards-compatible (although one could probably branch off a 2.x port with minimal effort).

**Warning**: ```nttc``` performs no custom error-handling, so make sure your inputs are formatted properly! If you have questions, please let me know via email.

## System requirements

* [tsm](https://github.com/dfreelon/TSM)
* [nltk](https://www.nltk.org/)
* networkx
* matplot
* pandas
* numpy
* emoji
* pprint
* gensim
* spacy
* re

## Installation
```pip install nttc```

## Under Development

- .ftree parsing is currently under development:
  - .batch_map() using the .ftree option.
  - Specifically working on the indices_getter(), so that the entire file is parsed and available to use.

## Example notebooks

- See the ```assets/examples``` folder for example uses.

## Distribution update terminal commands

<pre>
# Create new distribution of code for archiving
sudo python3 setup.py sdist bdist_wheel

# Distribute to Python Package Index
python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
</pre>