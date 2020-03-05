# Infomap Data-Processing Examples

## Process Infomap .ftree files into module communities and edge data per module community

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
# Take full listified .ftree file and write per Period per Module hubs as a Dict
new_dict = dict_with_edges
dh = nttc.infomap_hub_maker(new_dict, file_type='ftree', mod_sample_size=10, hub_sample_size=-1)
print(
    '2.\nSample hub: ',
    dh['1']['info_hub']['1'][:5]
)
```

Output:
```
2.
Sample hub:  [{'node': '1', 'name': 'username1', 'score': 0.156246}, {'node': '2', 'name': 'username2', 'score': 0.138213}, {'node': '3', 'name': 'username3', 'score': 0.00534793}, {'node': '4', 'name': 'username4', 'score': 5.96884e-05}, {'node': '5', 'name': 'username5', 'score': 5.59752e-05}]
```

```python
# Write edge and node lists per module: 
## (num of periods, num of modules, Dict of module data from infomap_hub_maker)
dict_full = nttc.networks_controller(10,10,dh)
```

Output as DataFrame:
```
directed_count	source	source_name	target	target_name
0	0.00808964	8	username2	1	username1
1	0.00648447	11	username4	1	username3
2	0.00527613	6	username6	1	username1
3	0.00361715	18	username8	1	username1
4	0.00356268	4	username10	1	username4
```

## Tally infomap scores per hub

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