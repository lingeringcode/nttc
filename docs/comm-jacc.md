# Score &amp; visualize community-hub likeness

These methods assume that you understand what your top targets mean within the scope of your research study and broader corpus. For example, if you are studying a rather homogenous and stable group(s) over time, then the top targets can begin to isolate who were important actors being targeted and discussed by these groups, etc. If the groups are heterogenous, then the top targeted maintains the status of who are important actors among these groups, but that doesn't mean that one community hub from period 1 is of similar homiphily as the targets in period 2, for example. Period one's community could be of a conservative partisanship, for instance, while period two's community could be of liberal partisanship.

## Analyze and return a list of alike communities across periods based on top targets:

1. Init a new ```matchingCommunitiesObject``` and write a dict of users with ```matching_dict_processor()``` to send to ```match_maker()```.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/matching_init_best_matches.png" />
2. Write a list of tuples (matched community pairs and their scores) with ```match_maker()``` to send to ```community_grouper()```.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/matching_sorted_filtered_comms.png" />
3. Analyze the intersections and unions of, in this case, the ```sorted_filtered_mentions```' values and output a list of sets, where each set includes alike communities across periods in the corpus.
    <img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/matching_groups.png" />

## Plot community similarity indices (Jaccard's Co-efficient)

<img src="https://github.com/lingeringcode/nttc/raw/master/assets/images/plot_comm_pairs.png" />

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
