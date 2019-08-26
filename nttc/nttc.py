# -*- coding: utf-8 -*-
#!/usr/bin/python3

# NTTC (Name That Twitter Community!) A Tweets Topic Modeling Processor for Python 3
# by Chris Lindgren <chris.a.lindgren@gmail.com>
# Distributed under the BSD 3-clause license.
# See LICENSE.txt or http://opensource.org/licenses/BSD-3-Clause for details.

# WHAT IS IT?
# A set of functions that process and create topic models from a sample of community-detected Twitter networks' tweets.
# It assumes you seek an answer to the following questions:
#    1.) What communities persist or are ephemeral across periods in the copora, and when?
#    2.) What can these communities be named, based on their sources, targets, topics, and top-RT'd tweets?
#    3.) Of these communities, what are their topics over time?
# Accordingly, it assumes you have a desire to investigate tweets from each detected community across
# already defined periodic episodes with the goal of naming each community AND examining their
# respective topics over time in the corpus.

# It functions only with Python 3.x and is not backwards-compatible (although one could probably branch off a 2.x port with minimal effort).

# Warning: nttc performs no custom error-handling, so make sure your inputs are formatted properly! If you have questions, please let me know via email.
from os import listdir
from os.path import join
import csv
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import functools
import operator
import re
import emoji
import string
import tsm

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
    See README.md for an overview aand comments for extended explanation.
'''

class communitiesObject:
    '''an object class with attributes for various matched-community data and metadata'''
    def __init__(self, tweet_slice=None, split_docs=None, id2word=None, texts=None,
                 corpus=None, readme=None, model=None, perplexity=None, coherence=None,
                 top_rts=None, top_mentions=None, full_hub=None):
        self.tweet_slice = tweet_slice
        self.split_docs = split_docs
        self.id2word = id2word
        self.texts = texts
        self.corpus = corpus
        self.readme = readme
        self.model = model
        self.perplexity = perplexity
        self.coherence = coherence
        self.top_rts = top_rts
        self.top_mentions = top_mentions
        self.full_hub = full_hub

class communityGroupsObject:
    '''an object class with attributes for various matched-community data and metadata'''
    def __init__(self, best_matches_mentions=None, sorted_filtered_comms=None, groups=None):
        self.best_matches_mentions = best_matches_mentions
        self.sorted_filtered_comms = sorted_filtered_comms
        self.groups = groups

'''
    Initialize communityGroupsObject
'''
def initializeCGO():
    return communityGroupsObject()

'''
    Load CSV data
'''
def get_csv(sys_path, __file_path__, dtype_dict):
    df_tw = pd.read_csv(join(sys_path, __file_path__),
                               delimiter=',',
                               dtype=dtype_dict,
                               error_bad_lines=False)
    return df_tw

'''
    Write CSV data
'''
def write_csv(dal, sys_path, __file_path__):
    dal.to_csv(join(sys_path, __file_path__),
                                sep=',',
                                encoding='utf-8',
                                index=False)

'''
    Writes all objects and their respective source/target information
    to a CSV of "hubs"
'''
def create_hub_csv_files(**kwargs):
    list_of_dfs = []
    for fb in kwargs['full_obj']:
        list_of_dfs.append(kwargs['full_obj'][fb].full_hub)
    df_merged = pd.concat(list_of_dfs, axis=0).reset_index(drop=True)
    if kwargs['drop_dup_cols'] == True:
        df_merged_drop_dup = df_merged.loc[:,~df_merged.columns.duplicated()]
        write_csv(df_merged_drop_dup, kwargs['sys_path'], kwargs['output_file'])
    else:
        write_csv(df_merged_drop_dup, kwargs['sys_path'], kwargs['output_file'])
    print('File written to system.')

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
def tm_maker(**kwargs):
    np.random.seed(kwargs['random_seed'])
    nltk.download('wordnet')
    for split in kwargs['split_comms']:
        # Create Dictionary
        id2word = corpora.Dictionary(kwargs['split_comms'][split].split_docs.tolist())
        kwargs['split_comms'][split].id2word = id2word

        # Create Texts
        texts = kwargs['split_comms'][split].split_docs.tolist()
        kwargs['split_comms'][split].texts = texts

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        kwargs['split_comms'][split].corpus = corpus

        # Human readable format of corpus (term-frequency)
        read_me = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
        kwargs['split_comms'][split].readme = read_me

        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=kwargs['num_topics'],
                                                   random_state=kwargs['random_state'],
                                                   update_every=kwargs['update_every'],
                                                   chunksize=kwargs['chunksize'],
                                                   passes=kwargs['passes'],
                                                   alpha=kwargs['alpha'],
                                                   per_word_topics=kwargs['per_word_topics'])

        kwargs['split_comms'][split].model = lda_model

        # Compute Perplexity
        perplexity = lda_model.log_perplexity(corpus)
        kwargs['split_comms'][split].perplexity = perplexity
        print('\n', split, ' Perplexity: ', perplexity)  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        kwargs['split_comms'][split].coherence = coherence_lda
        print('\n', split, ' Coherence Score: ', coherence_lda)

    print('\n Modeling complete.')
    return kwargs['split_comms']

'''
    Appends hubs' top RT'd tweets and usersto respective period and community object
        -- Args:
            Dataframe of hub targets,
            Dict of Objects with .sources,
            String of period number
        -- Returns: Dict Object with new .top_rts per Object
'''
def get_hubs_top_rts(**kwargs):
    dft = kwargs['dft']
    all_tweets = dft.loc[:, lambda dft: ['community', 'username', 'tweets', 'retweets_count']]
    for tfd in kwargs['tdo']:
        per_comm_rts = all_tweets[all_tweets['community'] == tfd]
        per_comm_rts.rename(columns={'username': 'top_rters'}, inplace=True)
        top_renamed_per_comm_rts = per_comm_rts[:kwargs['top_num']].reset_index(drop=True)
        kwargs['tdo'][tfd].top_rts = top_renamed_per_comm_rts

    return kwargs['tdo']

'''
    Appends hubs' top mentions data to respective period and community object
        -- Args:
            Dataframe of hub top mentions,
            Dict of Objects,
            String of column name for period,
            String of period number,
            String of column name for the community number
        -- Returns: Dict Object with new .targets per Object
'''
def get_hubs_top_mentions(**kwargs):
    for f in kwargs['dict_obj']:
        hub_comm = kwargs['hubs'][(kwargs['hubs'][kwargs['col_period']] == kwargs['pn']) & (kwargs['hubs'][kwargs['col_comm']] == f)]
        hub_comm = hub_comm.reset_index(drop=True)
        hub_comm.rename(columns={'username': 'top_mentions'}, inplace=True)
        kwargs['dict_obj'][f].top_mentions = hub_comm
    return kwargs['dict_obj']

'''
    Merges hubs' top RTs and targets data as a full list per Community
        -- Args:
        -- Returns:
'''
def merge_rts_mentions(fo):
    for f in fo:
        dfs = [df for df in [fo[f].top_mentions, fo[f].top_rts]]
        df_merged = pd.concat(dfs, axis=1).reset_index(drop=True)
        fo[f].full_hub = df_merged.reset_index(drop=True)
    return fo

'''
    Processes input dataframe of network community hubs for use in the tsm.match_communities() function
        -- Args: A dataframe with Period, Period_Community (1_0), and top mentioned (highest in-degree) users
        -- Returns: Dictionary of per Period with per Period_Comm hub values as lists:
            {'1': {'1_0': ['nancypelosi',
               'chuckschumer',
               'senfeinstein',
               'kamalaharris',
               'barackobama',
               'senwarren',
               'hillaryclinton',
               'senkamalaharris',
               'repadamschiff',
               'corybooker'],
               ...
               },
               ...
               '10': {'10_3': [...] }
            }
'''
def matching_dict_processor(**kwargs):
    full_dict = {}
    for index, row in kwargs['df'].iterrows():
        period_check = kwargs['df'].values[index][0]
        key_check = kwargs['df'].values[index][1]
        if index > 0 and (period_check in full_dict):
            if (kwargs['df'].values[index-1][1] == key_check) and (kwargs['df'].values[index-1][0] == period_check) and (key_check in full_dict[period_check]):
                # Update to full_dict[period_check][key_check]
                full_dict[period_check][key_check].append(kwargs['df'].values[index][2])
            elif  (kwargs['df'].values[index-1][1] == key_check) and (key_check not in full_dict[period_check]):
                # Create new key-value and update to full_dict
                full_dict[period_check].update( { key_check: [kwargs['df'].values[index][2]] } )
        elif index > 0 and (period_check not in full_dict):
            full_dict.update( {period_check: { key_check: [kwargs['df'].values[index][2]] } } )
        elif index == 0:
            full_dict.update( {period_check: { key_check: [kwargs['df'].values[index][2]] } } )

    if kwargs['match_obj'] is None:
        return full_dict
    elif kwargs['match_obj'] is not None:
        kwargs['match_obj'].best_matches_mentions = full_dict
        return kwargs['match_obj']

'''
    Takes period dict from matching_dict_processor() and submits to tsm.match_communities() method.
    Assigns, filters, and sorts the returned values into
        -- Args: Dictionary of per Period with per Period_Comm hub values as lists; filter_jacc threshold value (float) between 0 and 1.
        -- Returns: List of tuples: period_communityxperiod_community, JACC score
            [('1_0x4_0', 0.4286),
            ('1_0x2_11', 0.4615),
            ('1_0x3_5', 0.4615),
            ... ]
'''
def match_maker(**kwargs):
    pc_matching = {} # Assign with complete best matches
    for f1 in kwargs['full_dict']:
        for f2 in kwargs['full_dict']:
            if f1 != f2:
                # Runs similarity index (Jaccard's Co-efficient) on all period-comm combinations
                match = tsm.match_communities(kwargs['full_dict'][f1], kwargs['full_dict'][f2], weight_edges=False)
                the_key = f1 + 'x' + f2
                pc_matching.update({ the_key: match.best_matches })

    all_comm_scores = []
    for bmd in pc_matching:
        for b in pc_matching[bmd]:
            all_comm_scores.append( (b, pc_matching[bmd][b]) )

    sorted_all_comm_scores = sorted(all_comm_scores, key=lambda x: x[1])

    # Filter out low scores
    filtered_comm_scores = []
    for s in sorted_all_comm_scores:
        if s[1] > kwargs['filter_jacc']:
            filtered_comm_scores.append(s)

    sorted_filtered_comm_scores = sorted(filtered_comm_scores, key=lambda x: x[0][0], reverse=False)

    if kwargs['match_obj'] is None:
        return sorted_filtered_comm_scores
    elif kwargs['match_obj'] is not None:
        kwargs['match_obj'].sorted_filtered_comms = sorted_filtered_comm_scores
        return kwargs['match_obj']

'''
    Plot the community comparisons as a bar chart
    -- Args:
        ax=None # Resets the chart
        counter = List of tuples returned from match_maker(),
        path = String of desired path to directory,
        output = String value of desired file name (.png)
    - Returns: Nothing.

'''
def plot_bar_from_counter(**kwargs):
    if kwargs['ax'] is None:
        fig = plt.figure()
        kwargs['ax'] = fig.add_subplot(111)

    frequencies = []
    names = []

    for c in kwargs['counter']:
        frequencies.append(c[1])
        names.append(c[0])

    N = len(names)
    x_coordinates = np.arange(len(kwargs['counter']))
    kwargs['ax'].bar(x_coordinates, frequencies, align='center')

    kwargs['ax'].xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    kwargs['ax'].xaxis.set_major_formatter(plt.FixedFormatter(names))

    plt.xticks(range(N)) # add loads of ticks
    plt.xticks(rotation='vertical')

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2 # inch margin
    s = maxsize/plt.gcf().dpi*N+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=-0.15)
    plt.savefig(join(kwargs['path'], kwargs['output']))
    print('File ', kwargs['output'], ' saved to ', kwargs['path'])
    plt.show()

'''
    group_reader(): Takes the period_community pairs and appends to dict if intersections occur. However,
    the returned dict requires furter analysis and processing, due to unknown order and
    content from the sorted and filtered communities, which is why they are then sent to
    the final_grouper by community_grouper, after completion here.
        - Args: Accepts the initial group dict, which is cross-referenced by the pair of
            period_community values extracted via a regex expression.
        - Returns: A dict of oversaturated comparisons, which are sent to final_grouper()
            for final analysis, reduction, and completion.

'''
def group_reader(group_dict, m1, m2):
    for g in list(group_dict): # parse list of dicts

        # If m1 in list, but not m2, append m2.
        if (m1 in group_dict[g]['matches']) and (m2 not in group_dict[g]['matches']):
            group_dict[g]['matches'].append(m2)
            return group_dict

        # Else if m1 not in list, but m2 is, append m1.
        elif (m1 not in group_dict[g]['matches']) and (m2 in group_dict[g]['matches']):
            group_dict[g]['matches'].append(m1)
            return group_dict

    # Get last key value from groups dict, and add 1 to it
    all_keys = group_dict.keys()
    last_key = list(all_keys)[-1]
    last_g = int(last_key) + 1

    # If union exists, rewrite list with union
    group_dict.update( { str(last_g): { 'matches': [m1, m2] } } )
    return group_dict

'''
    final_grouper(): Takes the period_community dictionaries and tests for their intersections.
        Then, it takes any intersections and joins them with .union and appends them into a
        localized running list, which will all be accrued in a running master list of that community.
        From there, each community result will be sorted by their length in descending order.
        - Args: Accepts the group dict from group_reader().
        - Returns: A dict of all unique period_community elements (2 or more) found to be similar.

'''
def final_grouper(**kwargs):
    inter_groups = []
    final_groups = []
    # Check for intersections first,
    # If intersection, check unions and assign
    # Append list of unions to inter_groups
    for a in kwargs['all_groups']:
        inter = {}
        inter_list = []
        union = {}
        for b in kwargs['all_groups']:
            inter = set(kwargs['all_groups'][a]['matches']).intersection( set(kwargs['all_groups'][b]['matches']) )
            if len(inter) >= 1:
                union = set(kwargs['all_groups'][a]['matches']).union( set(kwargs['all_groups'][b]['matches']) )
                inter_list.append(union)
        inter_groups.append(inter_list)

    # Sort list by their length in descending order
    descending_inter_groups = sorted(inter_groups, key=len)

    # Append first item as key
    for linter in descending_inter_groups:
        if list(linter)[0] not in final_groups:
            final_groups.append(list(linter)[0])
    return final_groups

'''
    community_grouper(): Controller function for process to group together communities found to be similar
    across periods in the corpus. It uses the 1) group_reader() and 2) final_grouper()
    functions to complete this categorization process.
        - Args: Accepts the network object (net_obj) with the returned value from nttc.match_maker(),
            which should be saved as .sorted_filtered_comms property: a list of tuples with
            sorted and filtered community pairs and their score, but it only uses the
            community values.
        - Returns: A list of sets, where each set is a grouped recurrent community:
            For example, 1_0, where 1 is the period, and 0 is the designated community
            number.

'''
def community_grouper(**kwargs):
    groups = {}
    communities_across_periods = []
    i = 0
    # These 2 patterns find the parts of the keys
    regex1 = r"(\b\w{1,2}_[^x]{1,2})"
    regex2 = r"(([^x]{1,5}\b))"
    for fcs in kwargs['match_obj'].sorted_filtered_comms:
        # Parse comms into distinct strings for comparison
        match1 = re.findall(regex1, fcs[0])
        match2 = re.findall(regex2, fcs[0])
        communities_across_periods.append( (match1[0], match2[0][0]) )
        # If not values exist in groups, update
        if not groups:
            groups.update( {str(i): { 'matches': [ match1[0], match2[0][0] ] }} )
        # Else send matches to group_reader()
        else:
            group_reader(groups, match1[0], match2[0][0])
    fg = final_grouper(all_groups=groups)
    kwargs['match_obj'].groups = fg
    return fg
