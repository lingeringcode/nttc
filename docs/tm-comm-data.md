# Build topic models 

See the [example notebook]() to see how to build topic models per community-detected submodules and save all variables to respective object properties.

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
