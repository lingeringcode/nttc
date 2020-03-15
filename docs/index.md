# NTTC (Name That Twitter Community!): Process and analyze community-detected data
by Chris Lindgren <chris.a.lindgren@gmail.com>
Distributed under the BSD 3-clause license. See LICENSE.txt or http://opensource.org/licenses/BSD-3-Clause for details.

## Overview

A set of functions that process and create topic models from a sample of community-detected Twitter networks' tweets. It also analyzes if there are potential persistent community hubs (either/and by top mentioned or top RTers).

It assumes you seek an answer to the following questions:
1. What communities persist or are ephemeral across periods in the corpora, and when?
2. What can these communities be named, based on their top RTs and users, top mentioned users, as well as generated topic models?
3. Of these communities, what are their topics over time?
    - TODO: Build corpus of tweets per community groups across periods and then build LDA models for each set.

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
* tqdm
* sklearn
* MulticoreTSNE
* hdbscan
* seaborn
* stop_words

## Installation
```pip install nttc```

## Objects

```nttc``` initializes and uses the following objects:

### ```periodObject```

Object with properties that store per Community subgraph properties. Object properties as follows:

- ```.comm_nums```: List of retrieved community numbers from the imported nodes data
- ```.subgraphs_dict```: Dictionary of period's community nodes and edges data.

### ```communitiesObject```

Object with properties that generate topic model and also help you name them more easily. Object properties are as follows:

- ```.content_slice```: dict of a sample community's content segments
- ```.split_docs```: split version of sampled tweets
- ```.id2word```: dict version of split_docs
- ```.texts```: Listified version of sample
- ```.corpus```: List of sample terms with frequency counts
- ```.readme```: If desired, printout readable version
- ```.model```: Stores the LDA topic model object
- ```.perplexity```: Computed perplexity score of topic model
- ```.coherence```: Computed coherence score of topic model
- ```.top_rts```: Sample of top 10 Rters and RTs for the community
- ```.top_mentions```: Sample of top 10 people mentioned
- ```.full_hub```: Combined version of top_rts and top_mentions as a DataFrame
    
### ```communityGroupsObject```

Object with properties that analyze community likeness scores and then groups alike communities across periods. Object properties as follows:

- ```.best_matches_mentions```: Dictionary of per Period with per Period hub top_mentions (users) values as lists
- ```.best_matches_rters```: Dictionary of per Period with per Period hub top_rters (users) values as lists
- ```.sorted_filtered_comms```: List of tuples, where each tuple has 1) the tested pair of communities between 2 periods, and 2) their JACC score. Example: ```('1_0x4_0', 0.4286)```
- ```.groups_mentions```: A list of sets, where each set are alike mention groups across periods, based on your given JACC threshold:<pre>[{'1_8', '2_18'},
{'3_7', '4_2'},
{'7_11', '8_0'},
{'10_11', '4_14', '5_14', '6_7', '9_11'},
{'1_0', '2_11', '3_5', '4_0', '5_5', '6_12'},
{'10_10', '1_9', '2_3', '3_3', '4_6', '5_2', '6_3', '7_0', '8_2', '9_4'},
{'10_6', '1_2', '2_4', '3_4', '4_13', '5_6', '6_5', '7_4', '8_7', '9_0'},
{'10_0', '1_12', '2_6', '3_0', '4_5', '5_7', '6_6', '7_3', '8_9', '9_5'}]</pre>
- ```.groups_rters```: A list of sets, where each set are alike RTer groups across periods, based on your given JACC threshold:<pre>[{'1_8', '2_18', '5_14'},
{'10_20', '5_18'},
{'5_2', '6_3', '7_0'},
{'5_1', '7_1'},
{'10_12', '2_3', '3_13', '6_8', '7_5', '8_4', '9_1'}]</pre>

## General Functions

```nttc``` contains the following general functions:

* ```initializePO```: Initializes a periodObject().
* ```initializeCGO```: Initializes a communityGroupsObject().
* ```get_csv```: Loads CSV data as a pandas DataFrame.
* ```batch_csv```: Merge a folder of CSV files into either one allPeriodsObject that stores a dict of all network nodes and edges per Period, or returns only the aforementioned dict, if no object is passed as an arg.
* ```write_csv```: Writes DataFrame as a CSV file.

## [Infomap](https://www.mapequation.org/) Data-Processing Functions

```nttc``` contains the following functions to process data into a usable format for the [Infomap](https://www.mapequation.org/) network analysis system. 

For example, it takes an edge list with usernames (username1, username2), and it translates it into the necessary Pajek file format (.net).

### ```listify_unique_users```

Take edge list (List of lists [source, target]) and create a list of unique users.

### ```check_protected_dataype_names```

Verify that edge names don't conflict with Python protected datatypes. If they do, append 2 underscores to its end and log it.

### ```index_unique_users```

Take list of unique users and append IDs

### ```target_part_lookup```

Lookup target in unique list and return to netify_edges()

### ```write_net_dict```

Writes s Dict of vertices (nodes) and arcs (edges) in preperation for formatting it into the Pajek file format (.net). It returns a dictionary akin to the following:<pre>p_dict = {
        'vertices': verts, # A List of vertices (nodes) with an ID [1, user1]
        'arcs': arcs # A list of arcs (edges) [source, target]
    }</pre>


### ```vert_lookup```

Helper function for ```write_net_dict```. It finds the matching username and returns the period_based ID.

### ```netify_edges```

Accepts list of lists (edges) and replaces the usernames with their unique IDs. This prepares output for the infomap code system.

### ```write_net_txt```

Outputs .txt file with edges in a .net format for the [Infomap](https://www.mapequation.org/) system:
  <pre>source target [optional weight]
  1 2
  2 4
  2 8
  5 4
  ...</pre>

It also contains functions that enable you to isolate and output a CSV file with the hubs from each period. It does so with custom parsers for the infomap [```.map```](https://www.mapequation.org/code.html#Map-format) and [```.ftree```](https://www.mapequation.org/code.html#Map-format) file formats:

### ```read_map_or_ftree```

Helper function for infomap_hub_maker. Slices period's ```.map``` or ```.ftree``` into their line-by-line indices and returns a dict of those values for use.

### ```indices_getter```

Helper function for batch_map. Parses each line in the file and returns a list of lists, where each sublists is a line in the file.

### ```batch_map```

Retrieves all map files in a single directory. It assumes that you have only the desired files in said directory. Returns a dict of each files based on their naming scheme with custom regex pattern. Each key denotes the file and its values are list of lists, where each sublist is a lines in the file.

- ```regex```= Regular expression for filename scheme
- ```path```= String. Path for directory with .map or .ftree files

### ```networks_controller```

Uses Dict data structure hydrated from the following functions:

- .batch_map()
- .ftree_edge_maker(), and
- .infomap_hub_maker().

It appends node names to edge data and also creates a node list for each module.

- Args:
    - p_sample: Integer. Number of desired periods to sample.
    - m_sample: Integer. Number of desired modules to sample.
    - Dict. Output from batch_map(), ftree_edge_maker(), and
        infomap_hub_maker(), which includes.
        - DataFrame. Module edge data.
        - DataFrame. Module node data with names.
- Return:
    - dict_network: Appends more accessible edge and node data.

### ```network_organizer```

Organizes infomap .ftree network edge and node data into Dict.

- Args:
    - m_edges: DataFrame. Per period module edge data
    - m_mod: List of Dicts. Per period list of module data 
- Return:
    - return_dict: Dict. Network node and edge data with names:<pre>
        { 
            return_dict: {
                'nodes': DataFrame,
                'edges': Dataframe
            }
        }</pre>

### ```content_sampler```

Sample content in each period per module, based on map equation flow-based community detection.

- Args:
    - network: Dict. Each community across periods edge and node data.
    - corpus: DataFrame.
    - period_dates: Dict of lists.
    - sample_size: Integer.
    - random: Boolean. True pulls randomized sample. False pulls top x tweets.
- Return:
    - Dict of DataFrames. Sample of content in each module per period

### ```sample_getter```

Samples corpus based on module edge data from infomap data. **NOTE**: It currently assumes the following column types in this exact order:
  - 'id', 'date', 'user_id', 'username', 'tweet', 'mentions', 'retweets_count', 'hashtags', 'link'
  - **TODO: Change column lookup and appending process to be flexible for user's needs.
    
- Args:
  - sample_size: Integer. Number of edges to sample.
  - edges: List of Dicts. Edge data.
  - period_corpus: DataFrame. Content corpus to be sampled.
  - sample_type: String. Option for 
    - 'modules': Samples tweets based on community module source-target relations.
    - 'ht_groups': Samples tweets based on use of hashtags. Must provide list of strings.
  - user_threshold:
  - random: Boolean. True will randomly sample fully retrieved set of tweet content
  - ht_list: List of strings. If sampling via hashtag groups, then provide a list of the hashtags. Default is None.
- Return:
  - DataFrame. Sampled content, based on infomap module edges.

### ```infomap_edges_sampler```

Sample edges in each period per module, based on map equation flow-based community detection.

- Args:
    - network: Dict. Each module edges data across periods edge and node data.
    - sample_size: Integer.
    - column_name: String. Name of desired column to sample.
    - random: Boolean. True pulls randomized sample. False pulls top x tweets.
- Return:
    - Dict of DataFrames. Sample of content in each module per period

### ```ranker```

Appends rank and percentages at different aggregate levels.

- Args:
    - ```rank_type```= String. Argument option for type of ranking to conduct. Currently only per_hub.
    - ```tdhn```= Dict of corpus. Traverses the 'info_hub'
- Return
    - ```tdhn```= Updated 'info_hub' with 'percentage_total' per hub and 'spot' for each node per hub,
- TODO: Add per_party and per_hubname

### ```append_rank```

Helper function for ranker(). It appends the rank number for the 'spot' value.

### ```append_percentages```

Helper function for ranker(). Appends each node's total_percentage to the list

- Args:
    - ``rl``= List of lists. Ranked list of nodes per hub

### ```score_summer```

Tally scores from each module per period and append a score_total to each node instance per module for every period.

- Args:
    - ```dhn```= Dict of hubs returned from info_hub_maker

### ```get_period_flow_total```

Helper function for score_summer. Tallies scores per Period across hubs.

- Args:
    - ```lpt```= List. Contains hub totals per Period.
- Return
    - Float. Total flow score for a Period.

### ```get_score_total```

Helper function for score_summer. Tallies scores per Hub.

- Args:
    - ```list_nodes```= List of Dicts
- Return
    - ```total```= Float. Total flow score for a Hub.

### ```infomap_hub_maker```

Takes fully hydrated Dict of the ```map``` or ```ftree``` files and parses its Nodes into per Period and Module Dicts.

- Args: 
    - ```file_type```= String. 'map' or 'ftree' file type designation
    - ```dict_map```= Dict of map files
    - ```mod_sample_size```= Integer. Number of modules to sample
    - ```hub_sample_size```= Integer. number of nodes to sample for "hub" of each module
- Output:
    - ```dict_map```= Dict with new ```info_hub``` key hydrated with hubs

### ```output_infomap_hub```

Takes fully hydrated infomap dict and outputs it as a CSV file.

- Args: 
    - ```header```= column names for DataFrame and CSV; 
        - Assumes they're in order with period and hub in first and second position
    - ```dict_hub```= Hydrated Dict of hubs
    - ``path``= Output path
    - ``file``= Output file name

### ```sampling_module_hubs```

Compares hub set with tweet data to ultimately output sampled tweets with hub information.

* Args:
  * ```period_dates```: Dict of lists that include dates for each period of the corpus
  * ```period_check```: String for option: Check against 'single' or 'multiple'
  * ```period_num```: Integer. If period_check == 'single', provide integer of period number.
  * ```df_all_tweets```: Pandas DataFrame of tweets
  * ```df_hubs```: Pandas DataFrame of infomapped hubs
  * ```top_rts_sample```: Integer of desired sample size of sorted top tweets (descending order)
  * ```hub_sample```: Integer of desired sample size to output
  * ```columns```: List of column names; each as a String. **Must match column names from tweet and hub data sets
* Returns DataFrame of top sampled tweets

### ```add_infomap```: 

Helper function for ```sampling_module_hubs```. It cross-references the sampled.

* Args:
  * ```dft```: DataFrame of sampled tweet data
  * ```dfh```: Full DataFrame of hubs data
  * ```period_num```: Integer of particular period number
* Returns List of Dicts with hub and info_name mentions info

### ```batch_output_period_hub_samples```

Periodic batch output that saves sampled tweets as a CSV. Assumes successively numbered periods.

* Args:
  * module_output: DataFrame of tweet sample data per Period per Module
  * period_total: Interger of total number of periods
  * file_ext: String of desired filename extension pattern
  * period_path: String of desired path to save the files
* Returns nothing

## periodObject Functions

* ```get_comm_nums```: Filters unique community column values into List
* ```comm_sender``` and ```write_community_list```: These 2 functions create a dict of nodes and edges to be saved as a property, .subgraphs_dict, of a periodObject. It does so by:
  1. Creates a List of nodes per Community
  2. Creates a List of edges per Community
  3. Appends dict of these lists to comprehensive dict for the period.
  4. Appends this period dict to the period)bject property: .subgraphs_dict
  5. Returns the object.
* ```add_comm_nodes_edges```: Function to more quickly generate new networkX graph of specific comms in a period.
* ```add_all_nodes_edges```: Function to more quickly generate new networkX graph of all comms in a period.
* ```draw_subgraphs```: Draws subgraphs with networkX module, but can do so with multiple communities across periods.

## communitiesObject Functions

### ```create_hub_csv_files```

Writes all of the objects' top rt'd/mentions information as a CSV of "hubs".

### ```get_comm_nums```

Filters Dataframe column community values into a List.

### ```get_all_comms```

Slice the full set to community and their respective tweets.

- Args: 
    - ```dft```: Dataframe
    - ```col_community```: String. Column name for community
    - ```col_tweets```: String. Column name for tweet content

### ```comm_dict_writer```

Write per Community content segments into a dictionary.

- Args:
    - ```comm_list```= List of community numbers / labels
    - ```df_content```= DataFrame of data set in question
    - ```comm_col```= String of column name for community/module
    - ```content_col```= Sring of column name for content to parse and examine
    - ```sample_size_percentage```= Desired percentage to sample from full set
- Returns Dict of sliced DataFrames (value) as per their community/module (key)

### ```split_community_tweets```

Isolates community's content, then splits string into list of strings per Tweet preparing them for the topic modeling.

- Args: 
  - ```col_name```: String. Community label as String, 
  - ```dict_comm_obj```: Dict of community objects
  - ```sample_size_percentage```: Float. Between 0 and 1. 
- Returns as Dataframe of content for respective community

```clean_split_docs```

Removes punctuation, makes lowercase, removes stopwords, and converts into dataframe for topic modeling.

### ```tm_maker```

Creates data for TM and builds an LDA TM.

- Args: Pass many of the gensim LDATopicModel() object arguments here, plus some helpers. See their documentation for more details (https://radimrehurek.com/gensim/models/ldamodel.html).
  - random_seed: Integer. Value for randomized seed.
  - single: Boolean. True assumes only one period of data being evaluated.
  - split_comms: 
      - If 'single' False, Dict of objects with respective TM data.
      - If 'single' True, object with TM data
  - num_topics: Integer. Number of topics to produce (k value)
  - random_state: Integer. Introduce random runs.
  - update_every: Integer. "Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning."
  - chunksize: Integer. "Number of documents to be used in each training chunk."
  - passes: Integer. "Number of passes through the corpus during training."
  - alpha: String. Pass options available via gensim package
  - per_word_topics: Boolean. 
- Returns: Either updated Dict of objects, or single Dict. Now ready for visualization or printing.

### ```get_hubs_top_rts```

Appends hubs' top 10 RT'd tweets and usernames to respective period and community object.

- Args:
  - Dataframe of hub top mentions,
  - Dict of Objects with .top_rts,
  - String of period number
- Returns: Dict Object with new .top_rts per Object

### ```get_hubs_mentions```

Appends hubs' mentions data to respective period and community object.
  - Args:
    - Dataframe of hub mentions,
    - Dict of Objects,
    - String of column name for period,
    - String of period number,
    - String of column name for the community number
  - Returns: Dict Object with new .top_mentions per Object

### ```merge_rts_mentions```

Merges hubs' sources and mentions data as a full list per Community.

## communityGroupsObject Functions

### ```matching_dict_processor```

Processes input dataframe of network community hubs for use in the tsm.match_communities() function.

- Args: A dataframe with Period, Period_Community (1_0), and top mentioned (highest in-degree) users
- Returns: Dictionary of per Period with per Period_Comm hub values as lists:<pre>
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
        }</pre>

### ```match_maker```

Takes period dict from matching_dict_processor() and submits to tsm.match_communities() method. Assigns, filters, and sorts the returned values into a list or tuples with findings.

- Args: 
  - Dictionary of per Period with per Period_Comm hub values as lists; 
  - filter_jacc threshold value (float) between 0 and 1.
- Returns: List of tuples: period_communityxperiod_community, JACC score<pre>
        [('1_0x4_0', 0.4286),
        ('1_0x2_11', 0.4615),
        ('1_0x3_5', 0.4615),
        ... ]</pre>

### ```plot_bar_from_counter```

Plot the community comparisons as a bar chart.

- Args:
  - ```ax=None``` # Resets the chart
  - ```counter``` = List of tuples returned from match_maker(),
  - ```path``` = String of desired path to directory,
  - ```output``` = String value of desired file name (.png)
- Returns: Nothing.

### ```community_grouper()```

Controller function for process to group together communities found to be similar across periods in the corpus. It uses the 1) group_reader() and 2) final_grouper() functions to complete this categorization process.

- Args: Accepts the network object (net_obj) with the returned value from nttc.match_maker(), which should be saved as .sorted_filtered_comms property: a list of tuples with sorted and filtered community pairs and their score, but it only uses the community values.
- Returns: A list of sets, where each set is a grouped recurrent community: For example, 1_0, where 1 is the period, and 0 is the designated community number.<pre>
[{'1_8', '2_18'},
 {'3_7', '4_2'},
 {'7_11', '8_0'},
 {'10_11', '4_14', '5_14', '6_7', '9_11'},
 {'1_0', '2_11', '3_5', '4_0', '5_5', '6_12'},
 {'10_10', '1_9', '2_3', '3_3', '4_6', '5_2', '6_3', '7_0', '8_2', '9_4'},
 {'10_6', '1_2', '2_4', '3_4', '4_13', '5_6', '6_5', '7_4', '8_7', '9_0'},
 {'10_0', '1_12', '2_6', '3_0', '4_5', '5_7', '6_6', '7_3', '8_9', '9_5'}]
    </pre>
- <strong>NOTE</strong>: <em>This algorithm isn't perfect. It needs some refinement, since it may output some overlaps. However, it certainly filters down the potential persistent communities with either top_mentions or top_rters across periods, so it saves you some manual comparative analysis labor.</em>

### ```group_reader()```

Takes the period_community pairs and appends to dict if intersections occur. However, the returned dict requires furter analysis and processing, due to unknown order and content from the sorted and filtered communities, which is why they are then sent to the ```final_grouper``` by ```community_grouper```, after completion here.

- Args: 
  - Accepts the initial group dict, which is cross-referenced by the pair of period_community values extracted via a regex expression.
- Returns: A dict of oversaturated comparisons, which are sent to final_grouper() for final analysis, reduction, and completion.

### ```final_grouper()```

Takes the period_community dictionaries and tests for their intersections. Then, it takes any intersections and joins them with .union and appends them into a localized running list, which will all be accrued in a running master list of that community. From there, each community result will be sorted by their length in descending order.

- Args: Accepts the group dict from group_reader().
- Returns: A dict of all unique period_community elements (2 or more) found to be similar.

## Distribution update terminal commands

<pre>
# Create new distribution of code for archiving
sudo python3 setup.py sdist bdist_wheel

# Distribute to Python Package Index
python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
</pre>