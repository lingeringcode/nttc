# NTTC (Name That Twitter Community!) A Tweets Topic Modeling Processor for Python 3
by Chris Lindgren <chris.a.lindgren@gmail.com>
Distributed under the BSD 3-clause license. See LICENSE.txt or http://opensource.org/licenses/BSD-3-Clause for details.

## Overview

A set of functions that process and create topic models from a sample of community-detected Twitter networks' tweets.

It assumes you seek an answer to the following questions:
1. What communities persist or are ephemeral across periods in the copora, and when?
2. What can these communities be named, based on their sources, targets, topics, and top-RT'd tweets?
3. Of these communities, what are their topics over time?

Accordingly, it assumes you have a desire to investigate tweets from each detected community across already defined periodic episodes with the goal of naming each community AND examining their respective topics over time in the corpus.

It functions only with Python 3.x and is not backwards-compatible (although one could probably branch off a 2.x port with minimal effort).

**Warning**: ```nttc``` performs no custom error-handling, so make sure your inputs are formatted properly! If you have questions, please let me know via email.

## System requirements

* [nltk](https://www.nltk.org/)
* pandas
* numpy
* emoji
* pprint
* gensim
* spacy

## Installation
```pip install nttc```

## Functions

PyLimn contains the following functions:

* ```get_csv```: Loads CSV data as a pandas DataFrame.
* ```get_comm_nums```: Filters Dataframe column community values into a List.
* ```get_all_comms```: Slice the full set to community and their respective tweets. Arguments: Full dataframe, strings of column names for community and tweets.
* ```comm_dict_writer```: Writes per Community tweets into a dictionary.
* ```split_community_tweets```: Isolates community's tweets, then splits string into list of strings per Tweet preparing them for the topic modeling. Returns as Dataframe of tweets for resepective community.
* ```clean_split_docs```: Removes punctuation, makes lowercase, removes stopwords, and converts into dataframe for topic modeling.
* ```tm_maker```: Creates data for TM and builds an LDA TM. 
* ```get_hubs_sources```: TBA.
* ```print_keywords```: TBA. 

__Sample code__

```python
import nttc

data_path = '//Users/name/project/periods/top_rts/encoded'
__file__ = 'p6_comm_top500mentions_in_top10000_rts_count_uid.csv'

dtype_dict={
    'community': str,
    'tweets': str,
    'retweets_count': int,
    'link': str,
    'username': str,
    'user_id': int
}

# 1. Load CSV
df_tweets = nttc.get_csv(data_path, __file__, dtype_dict)

# 2. Get community numbers into a List
comm_list = nttc.get_comm_nums(df_tweets)

# 3. Write dictionary of tweets organized by per Community perspective
dict_all_comms = nttc.comm_dict_writer(comm_list, df_tweets, 'community', 'tweets')
# 4 . Process tweets for each community
split_dict_all_comms = nttc.split_community_tweets(dict_all_comms, 'tweets')
tms_full_dict = nttc.tm_maker(2018, split_dict_all_comms, 
                         num_topics=5,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True) #pass any of the following gensim LDATopicModel() object arguments here
```


__Sample Output from Above Code__ 

```
3  Perplexity:  -7.618915328673395

 3  Coherence Score:  0.3740323991406477

 5  Perplexity:  -7.749282621692275

 5  Coherence Score:  0.36001967258313305

 6  Perplexity:  -7.475628335657981

 6  Coherence Score:  0.32547481443269244

 7  Perplexity:  -7.264458923588148

 7  Coherence Score:  0.31947706630738704

 8  Perplexity:  -7.839326042415438

 8  Coherence Score:  0.31957579040223866

 9  Perplexity:  -7.670416717009498

 9  Coherence Score:  0.28534510836872357

 10  Perplexity:  -7.370800819131035

 10  Coherence Score:  0.34724361008183413

 12  Perplexity:  -6.9411620263614795

 12  Coherence Score:  0.397521213421681

 17  Perplexity:  -6.068761633181642

 17  Coherence Score:  0.44224500342072987

 27  Perplexity:  -6.345910693707283

 27  Coherence Score:  0.41525260201784386

 Modeling complete.
```

```python
split_dict_all_comms['10'].model

<gensim.models.ldamodel.LdaModel at 0x12daa5e80>
```