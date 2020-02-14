# Infomap Data-Processing Examples

## Parse Infomap .ftree files into module communities and edge data per module community

```python
import nttc

# 1. Retrieve directory of .ftree files and save each line of the file within a list of lists to per Period Dict
ftree_path = '../infomap/output/nets/ftree/ftree'

# regex is the file pattern in a dedicated directory, e.g., 
# # r"\d{1,2}" will match the '1' in p1_ftree.ftree
dict_map = nttc.batch_map(regex=r"\d{1,2}", path=ftree_path, file_type='ftree')

# Print sample ftree modules
print(
    '1.\nIndices: ',
    dict_map['1']['indices']['ftree_modules'],
    '\n\nFirst 5 file lines of module section: ',
    dict_map['1']['lines'][dict_map['1']['indices']['ftree_modules'][0]:5],
    '\n\n'
)

# Print sample ftree links
five = dict_map['1']['indices']['ftree_links']['1']['indices'][0]+5
print(
    '2.\nIndices for module 1 links: ',
    dict_map['1']['indices']['ftree_links']['1']['indices'],
    '\n\nFirst 5 lines of period 1, module 1 links section: ',
    dict_map['1']['lines'][dict_map['1']['indices']['ftree_links']['1']['indices'][0]:five],
    '\n\n'
)
```

Output:

```
1.
Indices:  [2, 50521] 

First 5 file lines of module section:  ['1:1 0.156246 "username1" 4', '1:2 0.138213 "username2" 294', '1:3 0.00534793 "username3" 533'] 


2.
Indices for module 1 links:  [50525, 95864] 

First 5 lines of period 1, module 1 links section:  ['2 1 0.00383033', '5 1 0.00319596', '1359 1 0.00299684', '1359 2 0.00298003', '28 1 0.0025742'] 
```

```python
# Check output
dict_map['1']['indices']['ftree_links']['1']
```

Output:
```python
{'exit_flow': '0.0',
 'indices': [50525, 95864],
 'num_children': '7362',
 'num_edges': '45339'}
```

```python
copy_dict_map = dict_map
# Process each period's module edge data and stores as a DataFrame.
dict_with_edges = nttc.ftree_edge_maker(copy_dict_map)
```

Output for 10 periods:
```
Processing edge data for period 1
Processing edge data for period 2
Processing edge data for period 3
Processing edge data for period 4
Processing edge data for period 5
Processing edge data for period 6
Processing edge data for period 7
Processing edge data for period 8
Processing edge data for period 9
Processing edge data for period 10
Processing complete!
```

```python
# Check sample of dataframe output
dict_with_edges['2']['indices']['ftree_links']['2']['df_edges'][:10]
```

Output (dataframe):
```
index source	target	directed_count
0	2	1	0.0146604
1	7	1	0.0069932
2	192	1	0.00081632
3	639	1	0.000395405
4	109	1	0.000299742
5	3	1	0.000294408
6	4	1	0.000266507
7	261	1	0.00022959
8	525	1	0.000165815
9	747	1	0.000146682
```

```python
copy_dict_with_edges = dict_with_edges
# Listify as one complete corpus with edges denoted with period and module info
list_edges = nttc.infomap_edge_data_lister(copy_dict_with_edges,10,10)
```

```python
# Test output
list_edges[:10]
```

Output:
```
[['1', '1', '2', '1', '0.00383033'],
 ['1', '1', '5', '1', '0.00319596'],
 ['1', '1', '1359', '1', '0.00299684'],
 ['1', '1', '1359', '2', '0.00298003'],
 ['1', '1', '28', '1', '0.0025742'],
 ['1', '1', '5', '2', '0.00210943'],
 ['1', '1', '167', '1', '0.00191508'],
 ['1', '1', '167', '5', '0.00178379'],
 ['1', '1', '10', '1', '0.00140345'],
 ['1', '1', '8', '1', '0.00131033']]
```

```python
# Export list_edges as a CSV file
import pandas as pd
from os import listdir
from os.path import join
import csv

columns = ['period','module','source','target','target_score']
df_edge_data = pd.DataFrame(list_edges, columns=columns)

df_edge_data.to_csv(join('../infomap/output/nets/ftree/csv', 'infomap_complete_edge_data.csv'),
                    sep=',',
                    encoding='utf-8',
                    index=False)
```

## Sample hubs

```python
# Take full listified .ftree file and write per Period per Module hubs as a Dict
new_dict = dict_with_edges
dh = nttc.infomap_hub_maker(new_dict, file_type='ftree', mod_sample_size=10, hub_sample_size=-1)
print(
    '\nSample hub: ',
    dh['1']['info_hub']['1'][:10]
)
```

Output
```
Sample hub:  [{'node': '4', 'name': 'username1', 'score': 0.156246}, {'node': '294', 'name': 'username2', 'score': 0.138213}, {'node': '533', 'name': 'username3', 'score': 0.00534793}, {'node': '9835', 'name': 'username4', 'score': 5.96884e-05}, {'node': '3641', 'name': 'username5', 'score': 5.59752e-05}, {'node': '3523', 'name': 'username6', 'score': 2.10157e-05}, {'node': '20242', 'name': 'username7', 'score': 2.10157e-05}, {'node': '25683', 'name': 'username8', 'score': 1.73383e-05}, {'node': '32006', 'name': 'username9', 'score': 1.68126e-05}, {'node': '4008', 'name': 'username10', 'score': 1.68126e-05}]
```

```python
dhn = dh
totals_dhn = nttc.score_summer(dhn, hub_sample_size=50)
# Updated hubs with scores
totals_dhn['1']['info_hub']['1'][:2]
```

Output:
```python
[{'name': 'username1',
  'node': '4',
  'score': 0.156246,
  'total_hub_flow_score': 0.30013800000000007,
  'total_period_flow_score': 0.5393100000000002},
 {'name': 'username2',
  'node': '294',
  'score': 0.138213,
  'total_hub_flow_score': 0.30013800000000007,
  'total_period_flow_score': 0.5393100000000002}]
```

```python
# Example process to append period and community module labels
tdhn = totals_dhn
for p in tdhn:
    for h in tdhn[p]['info_hub']:
        top_name = tdhn[p]['info_hub'][h][0]['name']
        for n in tdhn[p]['info_hub'][h]:
            n.update({'period': p})
            n.update({'community': h})

tdhn['1']['info_hub']['1'][:2]
```

Output:
```python
[{'community': '1',
  'name': 'username1',
  'node': '4',
  'period': '1',
  'score': 0.156246},
 {'community': '1',
  'name': 'username2',
  'node': '294',
  'period': '1',
  'score': 0.138213}]
```
## Process .ftree node and edge data

After running the following functions:

- .batch_map()
- .ftree_edge_maker(), and
- .infomap_hub_maker().

You can subsequently write and organize the edges and nodes for further network analysis and visualization.

```python
'''
    Positional arguments: 
        1. Number of desired periods to sample.
        2. Number of desired modules to sample.
        3. Dict. Output from batch_map(), ftree_edge_maker(), and
           infomap_hub_maker(), which includes.
           - DataFrame. Module edge data.
           - List of dicts. Module node data with names.
'''
dict_full = nttc.networks_controller(10,10,dh)
```

Output updates its progress:

```
Processing period 1
Module 1
Module 2
Module 3
Module 4
Module 5
Module 6
Module 7
Module 8
Module 9
Module 10
Processing period 2
Module 1
...
```

Test outputs:

```python
dict_full['network']['1']['10']['edges'][:5]
```
Output (dataframe sample):
```
	directed_count	source	source_name	target	target_name
0	2.10157e-05	18	username2	1	username1
1	2.10157e-05	41	username3	1	username1
2	2.10157e-05	269	username4	1	username1
3	2.10157e-05	6	username5	1	username1
4	1.68126e-05	45	username6	1	username1
```

```python
dict_full['network']['1']['10']['nodes'][:5]
```
Output (dataframe sample):
```
username
0	username1
2	username2
4	username3
6	username4
8	username5
```