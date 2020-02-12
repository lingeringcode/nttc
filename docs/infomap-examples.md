# Infomap Data-Processing Examples

## Sample ranked hubs

Create a sample of hubs ranked by their information flow scores:

```python
import nttc

# 1. Retrieve directory of .map files and save each line of the file within a list of lists to per Period Dict
map_path = '../infomap/output/nets/maps'
dict_map = nttc.batch_map(regex=r"\d{1,2}", path=map_path, file_type='map')

print(
    '1.\nIndices: ',
    dict_map['1']['indices'],
     dict_map['2']['indices'],
    '\n\nFirst 5 file lines: ',
    dict_map['1']['lines'][:5],
    '\n\n'
)

# 2. Take full listified .map file and write per Period per Module hubs as a Dict
new_dict = dict_map
dh = nttc.infomap_hub_maker(new_dict, file_type='map', mod_sample_size=10, hub_sample_size=-1)
print(
    '2.\nSample hub: ',
    dh['10']['info_hub']['1'][:10]
)

# 3. Updated hubs with scores
dhn = dh
totals_dhn = nttc.score_summer(dhn, hub_sample_size=5)
print(
    '\n3. Updated hubs with scores: ',
    totals_dhn['1']['info_hub']['1'][:20]
)

# 4. Ranked hubs with percentages
tdhn = totals_dhn
ranked_tdhn = nttc.ranker(tdhn, rank_type='per_hub')
print(
    '\n4. Ranked hubs with percentages:\n\n',
    ranked_tdhn['1']['info_hub']['1'][:20]
)

# 5. Output .map hubs as a CSV
header = ['period', 'info_module', 'node', 'name', 'score','total_hub_flow_score','total_period_flow_score','percentage_total','spot','top_name']
nttc.output_infomap_hub(
    header=header, 
    filtered_header_length=4,
    dict_hub=rtdhn, 
    path='../infomap/output/nets/maps/csv', 
    file='infomap_hubs_100_5.csv')

```
Output:
```
1.
Indices:  {'modules': [7, 5388], 'nodes': [5390, 55909], 'links': [55911, 86141]} {'modules': [7, 5100], 'nodes': [5102, 50554], 'links': [50556, 73404]} 

First 5 file lines:  ['# modules: 5382', '# modulelinks: 30231', '# nodes: 50520', '# links: 110923', '# codelength: 11.5932'] 


2.
Sample hub:  [{'node': '1', 'name': 'realdonaldtrump', 'score': 0.135007}, {'node': '2', 'name': 'speakerpelosi', 'score': 0.012912}, {'node': '3', 'name': 'gop', 'score': 0.008224}, {'node': '4', 'name': 'senschumer', 'score': 0.002308}, {'node': '5', 'name': 'themarkpantano', 'score': 1.8e-05}, {'node': '6', 'name': 'joelpollak', 'score': 1.8e-05}, {'node': '7', 'name': 'repkevinbrady', 'score': 1.6e-05}, {'node': '8', 'name': 'bike_at_w4', 'score': 1.3e-05}, {'node': '9', 'name': 'busyelves', 'score': 1.3e-05}, {'node': '10', 'name': 'jorgeramosnews', 'score': 1.3e-05}]

3. Updated hubs with scores:  [{'node': '1', 'name': 'realdonaldtrump', 'score': 0.088483, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645}, {'node': '2', 'name': 'dhsgov', 'score': 0.076302, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645}, {'node': '3', 'name': 'anncoulter', 'score': 0.002534, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645}, {'node': '4', 'name': 'patriotlexi', 'score': 4.2e-05, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645}, {'node': '5', 'name': 'realdrolmo', 'score': 1.5e-05, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645}, {'node': '6', 'name': 'sickoftheswamp', 'score': 1.5e-05}, {'node': '7', 'name': 'rippdemup', 'score': 1.5e-05}, {'node': '8', 'name': 'flying4jc', 'score': 1.5e-05}, {'node': '9', 'name': 'theresamechele', 'score': 1.5e-05}, {'node': '10', 'name': 'chadpergram', 'score': 1.5e-05}, {'node': '11', 'name': 'medicalellen', 'score': 1.5e-05}, {'node': '12', 'name': 'shelley2021', 'score': 1.4e-05}, {'node': '13', 'name': 'lisamei62', 'score': 1.4e-05}, {'node': '14', 'name': 'angeloraygomez', 'score': 1.4e-05}, {'node': '15', 'name': 'john_kissmybot', 'score': 1.4e-05}, {'node': '16', 'name': '1gigisims', 'score': 1.4e-05}, {'node': '17', 'name': 'betzi1l', 'score': 1.4e-05}, {'node': '18', 'name': 'safety_canada', 'score': 1.4e-05}, {'node': '19', 'name': 'jennyjlfortn632', 'score': 1.4e-05}, {'node': '20', 'name': 'michaelbeatty3', 'score': 1.4e-05}]

4. Ranked hubs with percentages:

 [{'node': '1', 'name': 'realdonaldtrump', 'score': 0.088483, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645, 'percentage_total': 1.0, 'spot': 1}, {'node': '2', 'name': 'dhsgov', 'score': 0.076302, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645, 'percentage_total': 0.3117281050451238, 'spot': 2}, {'node': '3', 'name': 'anncoulter', 'score': 0.002534, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645, 'percentage_total': 0.010352533592623309, 'spot': 3}, {'node': '4', 'name': 'patriotlexi', 'score': 4.2e-05, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645, 'percentage_total': 0.000171588954573867, 'spot': 4}, {'node': '5', 'name': 'realdrolmo', 'score': 1.5e-05, 'total_hub_flow_score': 0.24477100000007698, 'total_period_flow_score': 0.45748200000006645, 'percentage_total': 6.12817694906668e-05, 'spot': 5}, {'node': '6', 'name': 'sickoftheswamp', 'score': 1.5e-05}, {'node': '7', 'name': 'rippdemup', 'score': 1.5e-05}, {'node': '8', 'name': 'flying4jc', 'score': 1.5e-05}, {'node': '9', 'name': 'theresamechele', 'score': 1.5e-05}, {'node': '10', 'name': 'chadpergram', 'score': 1.5e-05}, {'node': '11', 'name': 'medicalellen', 'score': 1.5e-05}, {'node': '12', 'name': 'shelley2021', 'score': 1.4e-05}, {'node': '13', 'name': 'lisamei62', 'score': 1.4e-05}, {'node': '14', 'name': 'angeloraygomez', 'score': 1.4e-05}, {'node': '15', 'name': 'john_kissmybot', 'score': 1.4e-05}, {'node': '16', 'name': '1gigisims', 'score': 1.4e-05}, {'node': '17', 'name': 'betzi1l', 'score': 1.4e-05}, {'node': '18', 'name': 'safety_canada', 'score': 1.4e-05}, {'node': '19', 'name': 'jennyjlfortn632', 'score': 1.4e-05}, {'node': '20', 'name': 'michaelbeatty3', 'score': 1.4e-05}]

5. infomap_hubs_100_5.csv  written to  ../infomap/output/nets/maps/csv

```

## Sample tweets based on hubs

**Currently under development, so it is not available.**

Create samples of tweets based on the hubs from each detected community:

1. Create infomap hubs from ```.ftree``` files:
  <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/infomap_hub_processing.png" />
2. Create a sample of tweet data cross-referenced with the infomapped hub file:
  <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/infomap_hub_sampling.png" />
3. Output sample in a batch based on the number of periods:
  <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/infomap_hub_sampling_output.png" />
