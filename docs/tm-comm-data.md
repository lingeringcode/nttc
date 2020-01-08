# Build topic models 

Build topic models per community-detected submodules and save all variables to respective object properties.

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

After the above process, the ```tms_full_dict``` is then saturated with TM objects:

```python
{'1': <nttc.nttc.communitiesObject at 0x12d5a5e80>,
 '10': <nttc.nttc.communitiesObject at 0x13067d9b0>,
 '11': <nttc.nttc.communitiesObject at 0x13067db38>,
 '12': <nttc.nttc.communitiesObject at 0x13067d278>,
 '13': <nttc.nttc.communitiesObject at 0x13067d6a0>,
 '14': <nttc.nttc.communitiesObject at 0x108457198>,
 '15': <nttc.nttc.communitiesObject at 0x13067d3c8>,
 '2': <nttc.nttc.communitiesObject at 0x13067d0f0>,
 '3': <nttc.nttc.communitiesObject at 0x13067d828>,
 '4': <nttc.nttc.communitiesObject at 0x1084574a8>,
 '5': <nttc.nttc.communitiesObject at 0x13067de48>,
 '6': <nttc.nttc.communitiesObject at 0x13067dfd0>,
 '7': <nttc.nttc.communitiesObject at 0x13067d518>,
 '8': <nttc.nttc.communitiesObject at 0x108457320>,
 '9': <nttc.nttc.communitiesObject at 0x13067dcc0>}
```

Using some outputs from each object, you can visualize the topic models as an intertopic dsintance map:

```python
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

pyLDAvis.enable_notebook()
c1_vis = pyLDAvis.gensim.prepare(
    tms_full_dict['1'].model, 
    tms_full_dict['1'].corpus, 
    tms_full_dict['1'].id2word)
c1_vis
```

<img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/intertopic_distance_map.png" />

You can also add top mentioned users to each community's ```.top_mentions``` property:

<img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/add_mentions.png" />

Merge top RT and mentioned users (in-degree) information to each community's ```.full_hub``` property:

<img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/merge_mentions_top_rts.png" />

You can also output the entire hub as a CSV for closer analysis:

```python
nttc.create_hub_csv_files(
    full_obj=full_obj,
    sys_path=data_path,
    output_file='p1_full_hubs.csv',
    drop_dup_cols=True
)
```
