from setuptools import setup
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'nttc',
  packages = ['nttc'],
  version = '0.4.5.2',
  description = 'A set of functions that process and create topic models from a sample of community-detected Twitter networks\' tweets. It also analyzes if there are potential persistent community hubs (either/and by top mentioned or top RTers), and can process and visualize network data across periods and communities.',
  author = 'Chris A. Lindgren',
  author_email = 'chris.a.lindgren@gmail.com',
  long_description=long_description,
  long_description_content_type="text/markdown",
  url = 'https://github.com/lingeringcode/nttc/',
  download_url = 'https://github.com/lingeringcode/nttc/',
  install_requires = ['pandas', 'numpy', 'emoji', 'nltk', 'pprint', 'gensim', 'spacy', 'tsm','matplot', 'networkx'],
  keywords = ['data processing', 'topic modeling', 'naming network communities', 'network graphing'],
  classifiers = [],
  include_package_data=True
)
