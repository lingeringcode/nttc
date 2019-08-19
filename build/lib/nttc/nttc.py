# -*- coding: utf-8 -*-
#!/usr/bin/python3

# NTTC (Name That Twitter Community!) A Tweets Topic Modeling Processor for Python 3
# release 1 (09/01/2019)
# by Chris Lindgren <chris.a.lindgren@gmail.com>
# Distributed under the BSD 3-clause license. 
# See LICENSE.txt or http://opensource.org/licenses/BSD-3-Clause for details.

# WHAT IS IT?
# A set of functions that process and create topic models from a sample of community-detected Twitter networks' tweets.
# It assumes you seek an answer to the following questions: 
#    1.) What communities persist or are ephemeral across periods in the copora, and when?
#    2.) What can these communities be named, based on their sources, targets, topics, and top-RT'd tweets?
#    3.) Of these communities, what are their topics over time?
# Accordingly, it assumes you have a desire to investigate tweets from each detected community across already defined periodic episodes
# with the goal of naming each community AND examining their respective topics over time in the corpus.

# It functions only with Python 3.x and is not backwards-compatible (although one could probably branch off a 2.x port with minimal effort).

# Warning: nttc performs no custom error-handling, so make sure your inputs are formatted properly! If you have questions, please let me know via email.
from os import listdir
from os.path import join
import csv
import pandas as pd
import functools
import operator
import re
import emoji
import string

# Stopwords
# Import stopwords with nltk.
import nltk
from nltk.corpus import stopwords

# Topic-Modeling imports
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np

'''
    .full_comm_dict:
        -- args: ( Dataframe (community, tweets), List (comm numbers as Strings))
        -- returns:  dictionary of all communities in per Community
    
    .split_comm_tweets: 
        Retrieve community-specific tweets from full list and split each tweet per Word into a list of words 
        by using community dataframe as the passed parameter, e.g., get_community_tweets(c0).
        -- args: 
        -- returns:
    
    .clean_split_docs: 
        Takes split community tweets and removes punctuation, makes lowercase, removes stopwords, and converts 
        into dataframe for topic modeling by using the dataframe of split tweets as the passed parameter, 
        e.g., clean_split_docs(c0_processed_docs_pre_clean)
        -- args: 
        -- returns: 
    
    .topic_model: 
        Creates a topic model per Community.
        -- args: 
        -- returns:
    
    .tm_scores: 
        Calculate each TM scores
    
    .id2word
    .texts
    .corpus
    .readme
    .model
    .perplexity
    .coherence
    .sources
'''
class communitiesObject:
    '''an object class with attributes for various matched-community data and metadata'''
    # 'tweet_slice', 'split_docs', 'id2word', 'texts', 'corpus', 'readme', 'model', 'scores', 'sources'
    def __init__(self, tweet_slice=None, split_docs=None, id2word=None, texts=None, 
                 corpus=None, readme=None, model=None, perplexity=None, coherence=None, 
                 sources=None):
        self.tweet_slice = tweet_slice
        self.split_docs = split_docs
        self.id2word = id2word
        self.texts = texts
        self.corpus = corpus
        self.readme = readme
        self.model = model
        self.perplexity = perplexity
        self.coherence = coherence
        self.sources = sources
        
        
'''
    Load the CSV data
'''
def get_csv(sys_path, __file_path__, dtype_dict):
    df_tw = pd.read_csv(join(sys_path, __file_path__), 
                               delimiter=',',
                               dtype=dtype_dict,
                               error_bad_lines=False)
    return df_tw

'''
    Filters community column values into List
'''
def get_comm_nums(dft):
    # Get community numbers
    c_list = []
    for c in dft['community'].values.tolist():
        if not c_list:
            c_list.append(c)
        elif c not in c_list:
            c_list.append(c)

    return c_list

'''
    Slice the full set to community and their respective tweets
        # Takes full dataframe, strings of column names for community and tweets
'''
def get_all_comms(dft, col_community, col_tweets):
    acts = dft.loc[:, lambda dft: [col_community, col_tweets]] # all tweets per Community
    return acts

'''
    Write per Community tweets into a dictionary
        # Args: community list
'''
def comm_dict_writer(comm_list, dft, col_community, col_tweets):
    dict_c = {}
    for c in comm_list:
        out_comm_obj = communitiesObject()
        all_comms = get_all_comms(dft, col_community, col_tweets)
        df_slice = all_comms[all_comms['community'] == c]
        df_slice = df_slice.reset_index(drop=True)
        out_comm_obj.tweet_slice = df_slice
        dict_c.update( { c: out_comm_obj } )
    
    return dict_c


'''
    Isolates community's tweets, then splits string into list of strings per Tweet
        preparing them for the topic modeling
        
        Args: Community number as String, Dictionary of communities' tweets
        Returns as Dataframe of tweets for resepective community
'''
def split_community_tweets(dict_comm_tweets, col_name):
    for cdf in dict_comm_tweets:
        c_tweets = dict_comm_tweets[cdf].tweet_slice[col_name]
        # Split tweets; includes emoji support
        c_split = []
        for t in c_tweets.values.tolist():
            em_split_emoji = emoji.get_emoji_regexp().split(t)
            em_split_whitespace = [substr.split() for substr in em_split_emoji]
            em_split = functools.reduce(operator.concat, em_split_whitespace)
            # Append split tweet to list
            c_split.append(em_split)

        # Transform list into Dataframe
        df_documents = pd.DataFrame()
        for ts in c_split:
            df_documents = df_documents.append( {'tweet': ts}, ignore_index=True )

        # Transform into list of processed docs
        split_docs = df_documents['tweet']
        cleaned_split_docs = clean_split_docs(split_docs)
        dict_comm_tweets[cdf].split_docs = cleaned_split_docs
    print( ' \'processed_docs\': dataframe written for each community dictionary.' )
    return dict_comm_tweets

'''
    Removes punctuation, makes lowercase, removes stopwords, and converts into dataframe for topic modeling
'''

def clean_split_docs(pcpd):
    nltk.download('stopwords')
    stop = stopwords.words('english')
    stop = set(stop) # Transform as a set for efficiency later
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    c_no_puncs = []
    for st in pcpd.values.tolist():
        s_list = []
        for s in st:
            # pass the translator to the string's translate method
            s_list.append( s.translate(translator) )
        c_no_puncs.append(s_list)
    
    # Make all lowercase
    c_no_puncs_lower = []
    for cc in c_no_puncs:
        new_list = []
        for c in cc:
            new_str = c.lower()
            if new_str:
                new_list.append(new_str)
        c_no_puncs_lower.append(new_list)
    
    # Remove stopwords
    c_cleaned = []
    for a in c_no_puncs_lower:
        ctw = []
        for st in a:
            if st not in stop:
                ctw.append(st)
        c_cleaned.append(ctw)
    
    # Convert list to dataframe
    df_c_cleaned_up = pd.DataFrame({ 'tweet': c_cleaned })
    p_clean_docs = df_c_cleaned_up['tweet']
    
    return p_clean_docs

'''
    Functions to create data for TM
'''
def tm_maker(random_seed, split_dict_all_comms, num_topics, random_state, update_every, chunksize, passes, alpha, per_word_topics):
    np.random.seed(random_seed)
    nltk.download('wordnet')
    for split in split_dict_all_comms:
        # Create Dictionary
        id2word = corpora.Dictionary(split_dict_all_comms[split].split_docs.tolist())
        split_dict_all_comms[split].id2word = id2word
        
        # Create Texts
        texts = split_dict_all_comms[split].split_docs.tolist()
        split_dict_all_comms[split].texts = texts
        
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        split_dict_all_comms[split].corpus = corpus
        
        # Human readable format of corpus (term-frequency)
        read_me = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
        split_dict_all_comms[split].readme = read_me
        
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=num_topics, 
                                                   random_state=random_state,
                                                   update_every=update_every,
                                                   chunksize=chunksize,
                                                   passes=passes,
                                                   alpha=alpha,
                                                   per_word_topics=per_word_topics)
        
        split_dict_all_comms[split].model = lda_model
        
        # Compute Perplexity
        perplexity = lda_model.log_perplexity(corpus)
        split_dict_all_comms[split].perplexity = perplexity
        print('\n', split, ' Perplexity: ', perplexity)  # a measure of how good the model is. lower the better.
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        split_dict_all_comms[split].coherence = coherence_lda
        print('\n', split, ' Coherence Score: ', coherence_lda)
        
    print('\n Modeling complete.')
    return split_dict_all_comms

# Divvy up users from original dataframe
def get_hubs_sources(dft, tms_full_dict):
    all_sources = dft.loc[:, lambda dft: ['community', 'username', 'tweets', 'retweets_count']]
    for tfd in tms_full_dict:
        per_comm_sources = all_sources[all_sources['community'] == tfd]
        tms_full_dict[tfd].sources = per_comm_sources.values.tolist()[:10]
    
    return tms_full_dict

def print_keywords(cn, lda_model, corpus):
    print(cn)
    pprint(lda_model.print_topics())