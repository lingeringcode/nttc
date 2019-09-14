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

# It functions only with Python 3.x and is not backwards-compatible.

# Warning: nttc performs no custom error-handling, so make sure your inputs are formatted properly! If you have questions, please let me know via email.
from os import listdir
from os.path import isfile, join
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
import networkx as nx

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
class allPeriodsObject:
    '''an object class with an attribute dict that stores per Period network data of nodes and edges in respective Dataframes'''
    def __init__(self, all_period_networks_dict=None):
        self.all_period_networks_dict = all_period_networks_dict

class periodObject:
    '''an object class with attributes that store per Community subgraph properties'''
    def __init__(self, comm_nums=None, subgraphs_dict=None):
        self.comm_nums = comm_nums
        self.subgraphs_dict = subgraphs_dict

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
    def __init__(self, best_matches_mentions=None, best_matches_rters=None, sorted_filtered_mentions=None, sorted_filtered_rters=None, groups_mentions=None, groups_rters=None):
        self.best_matches_mentions = best_matches_mentions
        self.best_matches_rters = best_matches_rters
        self.sorted_filtered_mentions = sorted_filtered_mentions
        self.sorted_filtered_rters = sorted_filtered_rters
        self.groups_mentions = groups_mentions
        self.groups_rters = groups_rters

##################################################################

## General Functions

##################################################################

'''
    Initialize allPeriodsObject
'''
def initializeAPO():
    return allPeriodsObject()

'''
    Initialize periodObject
'''
def initializePO():
    return periodObject()

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
    batch_csv(): Merge a folder of CSV files into either one allPeriodsObject 
        that stores a dict of all network nodes and edges per Period, or returns 
        only the aforementioned dict, if no object is passed as an arg.
    Args:
        - path= String of path to directory with files
        - all_po= Optional instantiated allPeriodsObject
    Return: Either
        - Dict of per Period nodes and edges, or
        - allPeriodsObject with Dict stored as property
'''
def batch_csv(**kwargs):
    # Pattern for period number from filename
    re_period = r"(\d{1,2})"
    periods = []
    network_dicts = {}
    
    # Write list of periods
    for f in listdir(kwargs['path']):
        period_num = re.search(re_period, f)
        if period_num:
            if not periods:
                periods.append(period_num.group(0))
            elif period_num.group(0) not in periods:
                periods.append(period_num.group(0))
    
    # Listify files within path and ignore hidden files
    list_of_files = [f for f in listdir(kwargs['path']) if not f.startswith('.') and isfile(join(kwargs['path'], f))]
    
    # If period column exists, consolidate
    if 'p_col_exists' in kwargs:
        df_obj = pd.concat([pd.read_csv(file, index=False) for file in list_of_files])

        print(
        'Batch DF merge complete. First 5 rows:\n\n',
        df_obj[:5],
        '\n\nDF summary:\n\n',
        df_obj.describe()
        )
    # If period column doesn't exist, make it from filenames
    else:
        re_node = r"(node)"
        re_edge = r"(edge)"
        # Consolidate all CSV files into one Dataframe
        for a_file in list_of_files:
            new_df = pd.read_csv(join(kwargs['path'], a_file))
            node_check = re.search(re_node, a_file)
            edge_check = re.search(re_edge, a_file)
            period_num = re.search(re_period, a_file)
            if node_check:
                if len(network_dicts) == 0:
                    network_dicts.update({period_num.group(0): {'nodes': new_df}})
                elif period_num.group(0) not in network_dicts:
                    network_dicts.update({period_num.group(0): {'nodes': new_df}})
                elif period_num.group(0) in network_dicts:
                    network_dicts[period_num.group(0)].update({'nodes': new_df})
            elif edge_check:
                if len(network_dicts) == 0:
                    network_dicts.update({period_num.group(0): {'edges': new_df}})
                elif period_num.group(0) not in network_dicts:
                    network_dicts.update({period_num.group(0): {'edges': new_df}})
                elif period_num.group(0) in network_dicts:
                    network_dicts[period_num.group(0)].update({'edges': new_df})
        if 'all_po' in kwargs:
            kwargs['all_po'].all_period_networks_dict = network_dicts
            return kwargs['all_po']
        if 'all_po' not in kwargs:
            return network_dicts

'''
    write_csv(): Writes Dataframe input as an output CSV file.
'''
def write_csv(dal, sys_path, __file_path__):
    dal.to_csv(join(sys_path, __file_path__),
                                sep=',',
                                encoding='utf-8',
                                index=False)
    print(__file_path__, ' written to ', sys_path)

##################################################################

## Infomap Data-Processing Functions

##################################################################
'''
    vert_lookup: Helper function for write_net_dict. It finds the
        matching username and returns the period_based ID.
'''
def vert_lookup(a, u_list):
    for u in u_list:
        if a == u[2]:
            return u[0]    

'''
    write_net_dict: Writes s Dict of vertices (nodes) and
        arcs (edges) in the Pajek file format (.net).
'''
def write_net_dict(**kwargs):
    index=1
    verts = []
    arcs = []
    for v in kwargs['vertices']:
        user = '\"'+v[1]+'\"'
        if len(v) == 3:
            verts.append([index, user, v[3]])
        else:
            verts.append([index, user, v[1]])
        index = index + 1

    for e in kwargs['keyed_edges']:
        if len(e) == 3:
            print(e[0], e[1], e[3])
        else:
            source = vert_lookup(e[0], verts)
            target = vert_lookup(e[1], verts)
            arcs.append([source,target])
    p_dict = {
        'vertices': verts,
        'arcs': arcs
    }
    return p_dict

'''
    write_net_txt(): Outputs .net file Pajek format:
        node_id nodes [optional weight]
        1 "user1"
        2 "user2"
        ...
        source target [optional weight]
        1 2
        2 1
        ...
'''
def write_net_txt(**kwargs):
    with open(join(kwargs['net_path'], kwargs['net_output']), "a") as f:
        print( '*Vertices', len(kwargs['vertices']), file=f )
        for v in kwargs['vertices']:
            user = '\"'+v[1]+'\"'
            if len(v) == 4:
                print(v[0], user, v[3], file=f)
            else:
                print(v[0], v[1], file=f)
        
        print( '*Arcs', len(kwargs['keyed_edges']), file=f )
        for e in kwargs['keyed_edges']:
            if len(e) == 3:
                print(e[0], e[1], e[3], file=f)
            else:
                print(e[0], e[1], file=f)
    print('File', kwargs['net_output'], 'written to', kwargs['net_path'])

'''
    index_unique_users(): Take list of unique users and append IDs
'''
def index_unique_users(ul):
    index = 1
    new_un_list = []
    for un in ul:
        new_un_list.append( (index, un) )
        index = index + 1
    return new_un_list

'''
  check_protected_dataype_names: Verify that edge names don't conflict with Python protected datatypes.
      If they do, append 2 underscores to its end and log it.
'''
def check_protected_dataype_names(le):
    index = 0
    for e in le:
        if (str(e[0]) == 'nan' or str(e[0]) == 'None' or str(e[0]) == 'undefined' or str(e[0]) == 'null'):
            print('Source with encoded name found: ', e[0], ' at index ', index)
            e[0] = str(e[0])+'__'
            print('Renamed edge as ', e)
        elif (str(e[1]) == 'nan' or str(e[1]) == 'None' or str(e[1]) == 'undefined' or str(e[0]) == 'null'):
            print('Target with encoded name found: ', e[1], ' at index ', index)
            e[1] = str(e[1])+'__'
            print('Renamed edge as ', e)
        index=index+1
    return le

'''
    listify_unique_users(): Take edge list and create a list of unique users
'''
def listify_unique_users(**kwargs):
    user_list = []
    status_counter = 0
    index = 0
    print('Starting the listifying process. This may take some time, depending upong the file size.')
    for source, target in kwargs['df_edges']:
        if status_counter == 50000:
            print(index, 'entries processed.')
            status_counter = 0
        if (source not in user_list and target not in user_list):
            user_list.append(source)
            user_list.append(target)
        elif (source in user_list and target not in user_list):
            user_list.append(target)
        elif (source not in user_list and target in user_list):
            user_list.append(source)
        index = index + 1
        status_counter = status_counter + 1
    print('Listifying complete!')
    return user_list

'''
    target_part_lookup(): Lookup target in unique list and return
    to netify_edges()
'''
def target_part_lookup(nu_list, target):
    for n in nu_list:
        if n[1] == target:
            return n[0] #Return target

'''
    netify_edges(); Accepts list of lists (edges) and replaces the
        usernames with their unique IDs. This prepares output for the
        infomap code system.
'''
def netify_edges(**kwargs):
    position = 0
    status_counter = 0
    new_edges_list = []
    print('Started revising edges to net format. This may take some time, depending upon the file size.')
    for source in kwargs['list_edges']:
        new_edge = None
        for un in kwargs['unique_list']:
            if status_counter == 50000:
                print(position, 'entries processed.')
                status_counter = 0
            if source[0] == un[1]:
                src = un[0]
                tar = target_part_lookup(
                    kwargs['unique_list'],
                    kwargs['list_edges'][position][1]) #send target for retrieval
                new_edge = [src, tar]
        new_edges_list.append(new_edge)
        position = position + 1
        status_counter = status_counter + 1
    print('Finished revising the edge list.')
    return new_edges_list

'''
    read_map: Helper function for infomap_hub_maker. Slices period's .map
    into their line-by-line indices and returns a dict of those values for use.
'''
def read_map(**kwargs):
    lines = [line.rstrip('\n') for line in open(join(kwargs['path'], kwargs['file']))]
    return lines

'''
    indices_getter: Helper function for batch_map. Parses each line in the file
    and returns a list of lists, where each sublists is a line in the file.
'''
def indices_getter(lines):
    re_modules = r"\*Modules\s\d{1,}"
    re_nodes = r"\*Nodes\s\d{1,}"
    re_links = r"\*Links\s\d{1,}"
    index = 0
    mod_index = 0
    node_index = 0
    link_index = 0
    mod_list = []
    node_list = []
    link_list = []
    
    for l in lines:
        mod_match = re.match(re_modules, l)
        node_match = re.match(re_nodes, l)
        link_match = re.match(re_links, l)
        
        if mod_match != None:
            mod_index = index
        elif node_match != None:
            node_index = index
        elif link_match != None:
            link_index = index
            link_list.append([link_index+1, len(lines)-1])
        index = index + 1
    
    mod_list.append([mod_index+1, node_index-1])
    node_list.append([node_index+1, link_index-1])
    
    dict_indices = {
        'modules': mod_list[0],
        'nodes': node_list[0],
        'links': link_list[0]
    }

    return dict_indices

'''
    batch_map: Retrieves all map files in a single directory. It assumes
    that you have only the desired files in said directory. Returns a dict
    of each files based on their naming scheme with custom regex pattern.
    Each key denotes the file and its values are list of lists, where each sublist
    is a lines in the file.
'''
def batch_map(**kwargs):
    # Pattern for period number from filename
    re_period = kwargs['regex']
    periods = []
    map_dicts = {}
    
    # Write list of periods
    for f in listdir(kwargs['path']):
        period_num = re.search(re_period, f)
        if period_num[0]:
            if not periods:
                periods.append(period_num[0])
            elif period_num[0] not in periods:
                periods.append(period_num[0])
    
    # Listify files within path and ignore hidden files
    list_of_files = [f for f in listdir(kwargs['path']) if not f.startswith('.') and isfile(join(kwargs['path'], f))]
    
    # Write per Period dict with each file as List of lines to parse
    for f in list_of_files:
        period_num = re.search(re_period, f)
        p = period_num[0]
        lines = read_map(path=kwargs['path'], file=f)
        indices = indices_getter(lines)
        map_dicts.update({ p: 
                          {
                              'lines': lines,
                              'indices': indices
                          } 
                         })
    
    return map_dicts

'''
    infomap_hub_maker: Takes fully hydrated Dict of the map files
        and parses its Nodes into per Period and Module Dicts.
        - Args: 
            - dict_map= Dict of map files
'''
def infomap_hub_maker(dict_map):
    re_mod = r"\d{1,}\:"        #Module number
    re_node = r"\:\d{1,2}"      #Node number
    re_name = r"\"\S{1,}\""     #Node name
    re_score = r"\d{1}\.\d{1,}" #Node's flow score
    
    # Travers each period in dictionary
    for p in dict_map:
        dict_hubs = {}
        # Get indices for nodes in period's map
        start = dict_map[p]['indices']['nodes'][0]
        end = dict_map[p]['indices']['nodes'][1]
        
        # Traverse nodes in period's map
        for n in dict_map[p]['lines'][start:end]:
            mod_match = re.match(re_mod, n)
            node_match = re.findall(re_node, n)
            name_match = re.findall(re_name, n)
            score_match = re.findall(re_score, n)

            mod = mod_match[0][0:-1]
            hub_list = []
            
            # Retrieve first 15 modules
            if int(mod) == 16:
                # Update period with infomap hubs
                dict_map[p].update({
                    'info_hub': dict_hubs
                })
                break # All done
            elif mod not in dict_hubs:
                node = node_match[0][1]
                name = name_match[0][1:-1]
                score = score_match[0]
                dict_hubs.update({mod:
                    [{
                        'node': node,
                        'name': name,
                        'score': score
                     }]
                })
            elif mod in dict_hubs and len(dict_hubs[mod]) >= 1 and len(dict_hubs[mod]) < 10:
                node = node_match[0][1:]
                name = name_match[0][1:-1]
                score = score_match[0]
                hub_list = {
                    'node': node,
                    'name': name,
                    'score': score
                }
                dict_hubs[mod].append(hub_list)
        
    return dict_map

'''
    output_infomap_hub: Takes fully hydrated infomap dict and outputs it as a CSV file.
        - Args: 
            - header= column names for DataFrame and CSV
            - dict_hub= Hydrated Dict of hubs
            - path= Output path
            - file= Output file name
'''
def output_infomap_hub(**kwargs):
    hubs = []
    for ih in kwargs['dict_hub']:
        for i in kwargs['dict_hub'][ih]['info_hub']:
            for h in kwargs['dict_hub'][ih]['info_hub'][i]:
                hubs.append( [int(ih), int(i), h['name'], int(h['node']), float(h['score'])] ) 
    df_info_hubs = pd.DataFrame(hubs, columns=kwargs['header'])

    df_info_hubs.to_csv(join(kwargs['path'], kwargs['file']),
                                    sep=',',
                                    encoding='utf-8',
                                    index=False)
    print(kwargs['file'], ' written to ', kwargs['path'])

##################################################################

## periodObject Functions

##################################################################

'''
    get_comm_nums(): Filters community column values into List
        Args: 
            - period_obj= Instantiated periodObject()
            - dft_comm_col= Dataframe column of community values of nodes
        Returns: A List of unique community numbers (Strings) within the period
            - Either the periodObject() with the new property comm_nums, or
            - List of comm numbers as Strings
'''
def get_comm_nums(**kwargs):
    # Get community numbers
    c_list = []
    for c in kwargs['dft_comm_col'].values.tolist():
        if not c_list:
            c_list.append(c)
        elif c not in c_list:
            c_list.append(c)
    
    if kwargs['period_obj']:
        kwargs['period_obj'].comm_nums = c_list
        return kwargs['period_obj']
    elif not kwargs['period_obj']:
        return c_list

'''
    comm_sender && write_community_list functions create a dict of nodes 
    and edges to be saved as a property, .subgraphs_dict, of a periodObject.

    1. Creates a List of nodes per Community
    2. Creates a List of edges per Community
    3. Appends dict of these lists to comprehensive dict for the period.
    4. Appends this period dict to the period)bject property: .subgraphs_dict
    5. Returns the object.

    Args: 
        - comm_nums= List of community numbers from periodObject.comm_nums
        - period_obj= Instantiated periodObject()
        - nodes= Dataframe of community nodes with a column named 'community'
        - edges= Dataframe of community edges
    Returns:
        - periodObject() with the new property .subgraphs_dict
'''
def comm_sender(**kwargs):
    new_comm_dict = {}
    for a in kwargs['comm_nums']:
        cl = []
        cl = [a]
        print(cl)
        comm_nodes = pd.DataFrame()
        comm_nodes = kwargs['nodes'][kwargs['nodes'].community.isin(cl)]
        parsed_comm = write_community_list(comm_nodes, kwargs['edges'], a)
        new_comm_dict.update(parsed_comm)
    print(len(new_comm_dict))
    kwargs['period_obj'].subgraphs_dict = new_comm_dict
    return kwargs['period_obj']

def write_community_list(cn, df_edges, a):
    node_list = []
    edge_list = []
    
    for n in cn.values.tolist():
        node_list.append(n[0])
    
    print('Sample NODES: ', node_list[:5])
    
    for e in df_edges.values.tolist():
        # ONLY Comm SRC-->TARGETS
        for c in node_list:
            if e[0] == c:
                for cc in node_list:
                    if e[1] == cc:
                        edge_list.append( (e[0], e[1]) )
    
    print('Sample EDGES: ', edge_list[:5])
    dict_comm = {}
    dict_comm.update({a: { 'nodes': node_list, 'edges': edge_list }})
    return dict_comm

'''
    add_comm_nodes_edges(): Function to more quickly generate new networkX graph of
        specific comms in a period
    Args: 
        - Nodes: 
        - Newly instantiated networkX graph object
        - Edges: 
    Returns: networkX graph object with nodes and edges
'''
def add_comm_nodes_edges(**kwargs):
    for n in kwargs['comms_dict']['nodes']:
        kwargs['g'].add_node(n)
    for source, target in kwargs['comms_dict']['edges']:
        kwargs['g'].add_edge(source, target)
    return kwargs['g']

'''
    add_all_nodes_edges(): Function to more quickly generate new 
        networkX graph of all comms in a period
    Args: 
        - Nodes: 
        - Newly instantiated networkX graph object
        - Edges: 
    Returns: networkX graph object with nodes and edges
'''
def add_all_nodes_edges(**kwargs):
    for cd in kwargs['comms_dict']:
        for n in kwargs['comms_dict'][cd]['nodes']:
            kwargs['g'].add_node(n)
        for source, target in kwargs['comms_dict'][cd]['edges']:
            kwargs['g'].add_edge(source, target)
    return kwargs['g']

'''
    Draws subgraphs with networkX module
    Args:
        plt.figure():
            - figsize= Tuple of (width,height) Integers, e.g., (50,35) for matplot figure
        nx.draw():
            - with_labels= Boolean for labels option
            - font_weight= value for networkX option (see spec)
            - node_size= value for networkX option (see spec)
            - width: value for networkX option for edge width
        nx.draw_networkx_nodes() and nx.draw_networkx_edges():
            - period_comm_list= #List of tuples, where the first value
                is the period object, and the second the specified community:
                [(p1_obj, 8), (p2_obj, 18)]
            - node_size= #Integer
            - edge_color= #Hex color code
            - edge_width= #Integer
            - edge_alpha= #Float b/t 0 and 1
            - axis= #string 'on' or 'off' value
            - graph_titles= #List of Strings of desired titles for each 
                graph. Its order should follow period_comm_list[]
            - font_dict= #Dict with font options via matplot spec
            - output_paths= #List of Strings of desired paths and filenames 
                to save the image. Its order should follow period_comm_list[].
'''
def draw_subgraphs(**kwargs):
    period_index = 0
    # For each community list, draw the nodes and edges 
    # with specifying attributes
    for pc in kwargs['period_comm_list']:
        plt.figure(figsize=kwargs['figsize'])
        G = nx.DiGraph()
        if len(list(G.nodes())) > 0:
            G.clear() #Fresh graph
        else:
            G = add_comm_nodes_edges(comms_dict=pc[0].subgraphs_dict[pc[1]], g=G)
            pos_custom = nx.nx_agraph.graphviz_layout(G, prog=kwargs['graph_type'])

            nx.draw(G, pos_custom, with_labels=kwargs['with_labels'],
                font_weight=kwargs['font_weight'], node_size=kwargs['node_size'], width=kwargs['width'])

            node_list = pc[0].subgraphs_dict[pc[1]]['nodes']
            edge_list = pc[0].subgraphs_dict[pc[1]]['edges']
            # Draw nodes
            nx.draw_networkx_nodes(
                G,
                pos_custom,
                nodelist=node_list,
                #update existing dict, before calling this func
                node_color=pc[0].subgraphs_dict[pc[1]]['node_color'],
                node_size=kwargs['node_size'])
            # Draw edges
            nx.draw_networkx_edges(
                G,
                pos_custom,
                edgelist=edge_list,
                #update existing dict, before calling this func
                edge_color=pc[0].subgraphs_dict[pc[1]]['edge_color'],
                width=kwargs['edge_width'],
                alpha=kwargs['edge_alpha'])

            plt.axis(kwargs['axis'])
            plt.title(kwargs['graph_titles'][period_index], fontdict=kwargs['font_dict'])
            plt.savefig(kwargs['output_paths'][period_index]) # save as image
            plt.show()
            period_index = period_index + 1

##################################################################

## communitiesObject Functions

##################################################################

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
        -- Returns: Dict Object with new .top_mentions per Object
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

##################################################################

## communityGroupsObject Functions

##################################################################
'''
    Processes input dataframe of network community hubs for use in the tsm.match_communities() function
        -- Args: A dataframe with Period, Period_Community (1_0), and top mentioned (highest in-degree) users
        -- Returns: Dictionary of per Period with per Period_Comm hub values as lists:
            {'1': {'1_0': ['nancypelosi','chuckschumer','senfeinstein',
                'kamalaharris','barackobama','senwarren','hillaryclinton',
                'senkamalaharris','repadamschiff','corybooker'],
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
                if kwargs['top_mentions'] == True:
                    full_dict[period_check][key_check].append(kwargs['df'].iloc[index]['top_mentions'])
                elif kwargs['top_rters'] == True:
                    full_dict[period_check][key_check].append(kwargs['df'].iloc[index]['top_rters'])
            elif  (kwargs['df'].values[index-1][1] == key_check) and (key_check not in full_dict[period_check]):
                # Create new key-value and update to full_dict
                if kwargs['top_mentions'] == True:
                    full_dict[period_check].update( { key_check: [kwargs['df'].iloc[index]['top_mentions']] } )
                elif kwargs['top_rters'] == True:
                    full_dict[period_check].update( { key_check: [kwargs['df'].iloc[index]['top_rters']] } )
        elif index > 0 and (period_check not in full_dict):
            if kwargs['top_mentions'] == True:
                full_dict.update( {period_check: { key_check: [kwargs['df'].iloc[index]['top_mentions']] } } )
            elif kwargs['top_rters'] == True:
                full_dict.update( {period_check: { key_check: [kwargs['df'].iloc[index]['top_rters']] } } )
        elif index == 0:
            if kwargs['top_mentions'] == True:
                full_dict.update( {period_check: { key_check: [kwargs['df'].iloc[index]['top_mentions']] } } )
            elif kwargs['top_rters'] == True:
                full_dict.update( {period_check: { key_check: [kwargs['df'].iloc[index]['top_rters']] } } )

    if kwargs['match_obj'] is None:
        return full_dict
    elif kwargs['match_obj'] is not None:
        if kwargs['top_mentions'] == True:
            kwargs['match_obj'].best_matches_mentions = full_dict
        elif kwargs['top_rters'] == True:
            kwargs['match_obj'].best_matches_rters = full_dict
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
        if kwargs['top_mentions'] is True:
            kwargs['match_obj'].sorted_filtered_mentions = sorted_filtered_comm_scores
            return kwargs['match_obj']
        elif kwargs['top_rters'] is True:
            kwargs['match_obj'].sorted_filtered_rters = sorted_filtered_comm_scores
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
    # These 2 patterns find the parts of the keys: 
    # Example: 10_6x3_20 becomes 10_6 and 3_20, respectively
    regex1 = r"(\b\w{1,2}_[^x]{1,2})"
    regex2 = r"(([^x]{1,5}\b))"
    if kwargs['top_mentions'] is True:
        for fcs in kwargs['match_obj'].sorted_filtered_mentions:
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
        kwargs['match_obj'].groups_mentions = fg
        print('Be sure to double-check the output!')
        return kwargs['match_obj']
    elif kwargs['top_rters'] is True:
        for fcs in kwargs['match_obj'].sorted_filtered_rters:
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
        kwargs['match_obj'].groups_rters = fg
        print('Be sure to double-check the output!')
        return kwargs['match_obj']
