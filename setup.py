from setuptools import setup
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'nttc',
  packages = ['nttc'],
  version = '0.5.5',
  description = 'A set of functions that process and create topic models from a sample of community-detected Twitter networks\' tweets. It can process and visualize network data across periods and communities.',
  author = 'Chris A. Lindgren',
  author_email = 'chris.a.lindgren@gmail.com',
  long_description=long_description,
  long_description_content_type="text/markdown",
  url = 'https://github.com/lingeringcode/nttc/',
  download_url = 'https://github.com/lingeringcode/nttc/',
  install_requires = ['pandas', 'numpy', 'emoji', 'nltk', 'pprint', 'gensim', 'spacy', 'tsm', 'sklearn', 'MulticoreTSNE', 'hdbscan', 'seaborn', 'matplot', 'networkx', 'stop_words', 'tqdm'],
  keywords = ['data processing', 'topic modeling', 'naming network communities', 'network graphing'],
  classifiers = [],
  include_package_data=True
)
