# Sample community-detected data

Suppose you would like to sample your data to either topic-model or qualitatively code. See the [example notebook](https://github.com/lingeringcode/nttc/tree/master/assets/examples) to sample tweets based on community module or hashtag groupings. If you are using .ftree files, then follow the provided importing and process. If using another format, then you will need to process the edge data as a ```Dict```:

```python
'dict_full':{
  'network':{
    '1': {
      '1':
    },
    ...
  },
  ...
}
```

