# Hate Speech in Social Media Project


## Setup

This repository requires python 3.7.3, pip and virtualenv. Setup a virtual environment as follows:

```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Workflow

### YouTube data environment variable

Ping @danielwatson6 for access to the YouTube corpus. Set up an environment variable `DATASETS=/path/to/youtube/data/dir` and name the directory with the CSV files `youtube_right`.

### Python

Whenever working with python, run `source env/bin/activate` in the current terminal session. If new packages are installed, update the list of dependencies by running `pip freeze > requirements.txt`.

### Embedding visualization

After running a script that produces visualizations (for example, `centroids.py`), go to [projector.tensorflow.org](http://projector.tensorflow.org) and upload the TSV files inside the `projector` directory.

## Datasets

- [Stormfront dataset](https://github.com/aitor-garcia-p/hate-speech-dataset): place the `all_files` directory and the `annotations_metadata.csv` file inside this repository's `data` directory. Rename `all_files` to `stormfront`, and `annotations_metadata.csv` to `stormfront.csv`.

- [Twitter hate speech dataset](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv?raw=true): rename the file to `twitter.csv` and place it in the `data` directory.

- [Google News Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing): place the file directly in the `data` directory.

### Scraping YouTube

For the scraping script to work, place your YouTube API key in a file `scraping/api_key`.

