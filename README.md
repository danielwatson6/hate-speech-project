# Hate Speech in Social Media Project

## Setup

This repository requires python 3.7.3, pip and virtualenv. Setup a virtual environment as follows:

```bash
virtualenv env
source env.sh
pip install -r requirements.txt
```

## Workflow

### Python

Whenever working with python, run `source env.sh` in the current terminal session. If new packages are installed, update the list of dependencies by running `pip freeze > requirements.txt`.

Scripts in subdirectories (e.g. those in `scraping`) should be run as modules to avoid path and module conflicts:

```bash
python -m scraping.new_dataset  # instead of `python scraping/new_dataset.py` or `cd scraping && python new_dataset.py`
```

### TensorFlow

The TensorFlow workflow in this repository is adapted from [this boilerplate](https://github.com/danielwatson6/tensorflow-boilerplate).

#### Model visualization

During and after training, the training and validation losses are plotted in TensorBoard. To visualize, run `tensorboard --logdir=experiments` and open [localhost:6006](localhost:6006) in the browser.

#### Embedding visualization

After running a script that produces visualizations (for example, `scripts.centroids`), go to [projector.tensorflow.org](http://projector.tensorflow.org) and upload the TSV files inside the `projector` directory.

## Datasets

- [Stormfront dataset](https://github.com/aitor-garcia-p/hate-speech-dataset): place the `all_files` directory and the `annotations_metadata.csv` file inside this repository's `data` directory. Rename `all_files` to `stormfront`, and `annotations_metadata.csv` to `stormfront.csv`.

- [Twitter hate speech dataset](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv?raw=true): rename the file to `twitter.csv` and place it in the `data` directory.

- [Google News Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing): place the file directly in the `data` directory.

- [Twitter moral foundations dataset](https://psyarxiv.com/w4f72/): rename the directory to `twitter_mf` and place it in the `data` directory. To scrape the tweets from their id's, run `python -m scraping.twitter_mf` and then to clean the data run `python -m scripts.clean_twitter_mf`. To have a fixed heldout dataset that represents well the rest of the data, create a shuffled version of the data:
```bash
cat data/twitter_mf.clean.csv | head -1 > data/twitter_mf.clean.shuffled.csv
# macOS users: `sort` by hash is a good replacement for `shuf`.
cat data/twitter_mf.clean.csv | tail -24771 | shuf >> data/twitter_mf.clean.shuffled.csv
```

- [WikiText](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/): download, unzip, and place both of the word-level datasets in the `data` directory. Clean the data with `python -m scripts.clean_wikitext`.

### Non-public data

Ping @danielwatson6 for access to the YouTube and the ambiguity corpora.
- Set up an environment variable `DATASETS=/path/to/youtube/data/dir` and name the directory with the YouTube CSV files `youtube_right`. This is done unlike with the rest of the data to avoid the massive dataset not fitting on available SSD space.
- Run `python -m scraping.new_dataset` to scrape the rest of the YouTube data.
- Rename the ambiguity data to `ambiguity.csv` and place it in the `data` folder.

### Scraping

For the scraping scripts to work, you need your own API keys.

- Place your YouTube API key in a file `scraping/api_key`.
- Place your Twitter API keys in a JSON file `scraping/twitter_api.json` with the following keys: `consumer_key`, `consumer_secret`, `access_token_key`, `access_token_secret`.
