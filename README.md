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

* **IMPORTANT**: [tsm](https://github.com/dfreelon/TSM) - Current version on Github, not Python Package Index, so you will need to manually download and install from Github (as of 08/23/19).
  1. Download the repo from the link above to your computer.
  2. Open a Terminal and change the directory to ```TSM-master```: ```cd path/to/TSM-master```
  3. Once the Terminal is in the root of the ```TSM-master``` folder, be sure to first verify that that tsm is not already installed on your computer: ```sudo pip3 uninstall tsm```. 
     - NOTE: If it is not installed, it will tell you. If it is installed, it will uninstall it. Also, if you have permissions on this folder, there's no need to use ```sudo```.
  4. Still inside of this root folder, ```TSM-master```, install this version: ```sudo pip3 install .```
  5. ```pip``` will tell you if the package has successfully been installed or not. If not, research the error.
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

## Objects

```nttc``` initializes and uses the following objects:

* ```periodObject```: Object with properties that store per Community subgraph properties. Object properties as follows:
    - ```.comm_nums```: List of retrieved community numbers from the imported nodes data
    - ```.subgraphs_dict```: Dictionary of period's community nodes and edges data.

* ```communitiesObject```: Object with properties that generate topic model and also help you name them more easily. Object properties are as follows:
    - ```.tweet_slice```: dict of a sample community's tweets
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
    
* ```communityGroupsObject```: Object with properties that analyze community likeness scores and then groups alike communities across periods. Object properties as follows:
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

```nttc``` contains the following functions to process data into a usable format for the [Infomap](https://www.mapequation.org/) network analysis system. In short, it takes an edge list with usernames (username1, username2), and it translates it into the necessary Pajek file format (.net).

* ```listify_unique_users```: Take edge list and create a list of unique users
* ```check_protected_dataype_names```: Verify that edge names don't conflict with Python protected datatypes. If they do, append 2 underscores to its end and log it.
* ```index_unique_users```: Take list of unique users and append IDs
* ```target_part_lookup```: Lookup target in unique list and return to netify_edges()
* ```write_net_dict```: Writes s Dict of vertices (nodes) and arcs (edges) in preperation for formatting it into the Pajek file format (.net). It returns a dictionary akin to the following:<pre>p_dict = {
        'vertices': verts, # A List of vertices (nodes) with an ID [1, user1]
        'arcs': arcs # A list of arcs (edges) [source, target]
    }</pre>
* ```vert_lookup```: Helper function for ```write_net_dict```. It finds the matching username and returns the period_based ID.
* ```netify_edges```: Accepts list of lists (edges) and replaces the usernames with their unique IDs. This prepares output for the infomap code system.
* ```write_net_txt```: Outputs .txt file with edges in a .net format for the [Infomap](https://www.mapequation.org/) system:
  <pre>source target [optional weight]
  1 2
  2 4
  2 8
  5 4
  ...</pre>

It also contains functions that enable you to isolate and output a CSV file with the hubs from each period. It does so with custom parsers for the infomap ```.map``` file formats:

* ```read_map```: Helper function for infomap_hub_maker. Slices period's ```.map``` into their line-by-line indices and returns a dict of those values for use.
* ```indices_getter```: Helper function for batch_map. Parses each line in the file and returns a list of lists, where each sublists is a line in the file.
* ```batch_map```: Retrieves all map files in a single directory. It assumes that you have only the desired files in said directory. Returns a dict of each files based on their naming scheme with custom regex pattern. Each key denotes the file and its values are list of lists, where each sublist is a lines in the file.
* ```infomap_hub_maker```: Takes fully hydrated Dict of the map files and parses its Nodes into per Period and Module Dicts.
  * Args:
    * dict_map= Dict of map files
* ```output_infomap_hub```: Takes fully hydrated infomap dict and outputs it as a CSV file.
  * Args:
    * header= column names for DataFrame and CSV
    * dict_hub= Hydrated Dict of hubs
    * path= Output path
    * file= Output file name

Here's an example use of the above functions for creating infomap hubs from ```.map``` files:

<img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/infomap_hub_processing.png" />

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

* ```create_hub_csv_files```: Writes all of the objects' top rt'd/mentions information as a CSV of "hubs"
* ```get_comm_nums```: Filters Dataframe column community values into a List.
* ```get_all_comms```: Slice the full set to community and their respective tweets. Arguments: Full dataframe, strings of column names for community and tweets.
* ```comm_dict_writer```: Writes per Community tweets into a dictionary.
* ```split_community_tweets```: Isolates community's tweets, then splits string into list of strings per Tweet preparing them for the topic modeling. Returns as Dataframe of tweets for resepective community.
* ```clean_split_docs```: Removes punctuation, makes lowercase, removes stopwords, and converts into dataframe for topic modeling.
* ```tm_maker```: Creates data for TM and builds an LDA TM.
* ```get_hubs_top_rts```: Appends hubs' top 10 RT'd tweets and usernames to respective period and community object.
  - Args:
    - Dataframe of hub top mentions,
    - Dict of Objects with .top_rts,
    - String of period number
  - Returns: Dict Object with new .top_rts per Object
* ```get_hubs_mentions```: Appends hubs' mentions data to respective period and community object.
  - Args:
    - Dataframe of hub mentions,
    - Dict of Objects,
    - String of column name for period,
    - String of period number,
    - String of column name for the community number
  - Returns: Dict Object with new .top_mentions per Object
* ```merge_rts_mentions```: Merges hubs' sources and mentions data as a full list per Community.

## communityGroupsObject Functions

* ```matching_dict_processor```: Processes input dataframe of network community hubs for use in the tsm.match_communities() function.
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
* ```match_maker```: Takes period dict from matching_dict_processor() and submits to tsm.match_communities() method. Assigns, filters, and sorts the returned values into
    - Args: Dictionary of per Period with per Period_Comm hub values as lists; filter_jacc threshold value (float) between 0 and 1.
    - Returns: List of tuples: period_communityxperiod_community, JACC score<pre>
            [('1_0x4_0', 0.4286),
            ('1_0x2_11', 0.4615),
            ('1_0x3_5', 0.4615),
            ... ]</pre>
* ```plot_bar_from_counter```: Plot the community comparisons as a bar chart.
    - Args:
      - ax=None # Resets the chart
      - counter = List of tuples returned from match_maker(),
      - path = String of desired path to directory,
      - output = String value of desired file name (.png)
    - Returns: Nothing.
* ```community_grouper()```: Controller function for process to group together communities found to be similar across periods in the corpus. It uses the 1) group_reader() and 2) final_grouper() functions to complete this categorization process.
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
* ```group_reader()```: Takes the period_community pairs and appends to dict if intersections occur. However, the returned dict requires furter analysis and processing, due to unknown order and content from the sorted and filtered communities, which is why they are then sent to the final_grouper by community_grouper, after completion here.
    - Args: Accepts the initial group dict, which is cross-referenced by the pair of period_community values extracted via a regex expression.
    - Returns: A dict of oversaturated comparisons, which are sent to final_grouper() for final analysis, reduction, and completion.
* ```final_grouper()```: Takes the period_community dictionaries and tests for their intersections. Then, it takes any intersections and joins them with .union and appends them into a localized running list, which will all be accrued in a running master list of that community. From there, each community result will be sorted by their length in descending order.
    - Args: Accepts the group dict from group_reader().
    - Returns: A dict of all unique period_community elements (2 or more) found to be similar.



__Build a topic model per Community and save all variables to respective object properties.__

```python
import nttc

data_path = '//Users/name/project/periods/top_rts/encoded'
__file__ = 'p1_comm_top500mentions_in_top10000_rts_count_uid.csv'

dtype_dict={
    'community': str,
    'tweets': str,
    'retweets_count': float,
    'link': str,
    'username': str,
    'user_id': float
}

# 1. Load CSV
df_tweets = nttc.get_csv(data_path, __file__, dtype_dict)
# 2. Get community numbers into a List
comm_list = nttc.get_comm_nums(df_tweets)
# 3. Write dictionary of tweets organized by per Community perspective
dict_all_comms = nttc.comm_dict_writer(comm_list, df_tweets, 'community', 'tweets')
# 4 . Process tweets for each community
split_dict_all_comms = nttc.split_community_tweets(dict_all_comms, 'tweets')
# 5. Build the topic model
tms_full_dict = nttc.tm_maker(random_seed=2018,
                                                    split_comms=split_dict_all_comms,
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

**Add top mentioned users to each community's ```.top_mentions``` property**

<img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/add_mentions.png" />

**Merge top RT and mentioned users (in-degree) information to each community's ```.full_hub``` property**

<img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/merge_mentions_top_rts.png" />

**Using some outputs from each object, you can visualize the topic models**

<img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/intertopic_distance_map.png" />

**You can also output the entire hub as a CSV for closer analysis**

```python
nttc.create_hub_csv_files(
    full_obj=full_obj,
    sys_path=data_path,
    output_file='p1_full_hubs.csv',
    drop_dup_cols=True
)
```

**Plot community similarity indices (Jaccard's Co-efficient)**

<img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/plot_comm_pairs.png" />

**Analyze and return a list of alike communities across periods**

1. Init a new ```matchingCommunitiesObject``` and write a dict of users with ```matching_dict_processor()``` to send to ```match_maker()```.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/matching_init_best_matches.png" />
2. Write a list of tuples (matched community pairs and their scores) with ```match_maker()``` to send to ```community_grouper()```.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/matching_sorted_filtered_comms.png" />
3. Analyze the intersections and unions of, in this case, the ```sorted_filtered_mentions```' values and output a list of sets, where each set includes alike communities across periods in the corpus.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/matching_groups.png" />

**Process and visualize alike communities across periods**

1. Init a new ```allPeriodsObject``` and merge all network node and edge data from each period into one dict with ```batch_csv()```.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/all_periods_import.png" />
2. Init new periodObj() as needed with ```initializePO()```, retrieve the unique comunity numbers with ```get_comm_nums()```, and slice up each period per Community into the periodObject with ```comm_sender()```. Repeat per Period.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/all_periods_slice_comms_into_per_objs.png" />
3. If desired, and as needed, access each periods network data for use.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/all_periods_comms_access.png" />
4. Visualize each alike community network graph with ```draw_subgraphs()```, a networkx helper function, as desired.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/all_periods_draw_subgraphs_code.png" />
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/all_periods_draw_subgraphs_output.png" />

## Distribution update terminal commands

<pre>
# Create new distribution of code for archiving
sudo python3 setup.py sdist bdist_wheel

# Distribute to Python Package Index
python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
</pre>