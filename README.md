# Hate Speech in Social Media Project


## Setup

This repository requires python 3.7.3, pip and virtualenv. Setup a virtual environment as follows:

```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Workflow

Whenever working with python, run `source env/bin/activate` in the current terminal session. If new packages are installed, update the list of dependencies by running `pip freeze > requirements.txt`.


## Datasets

Download the [Stormfront dataset](https://github.com/aitor-garcia-p/hate-speech-dataset) and place the `all_files` directory and the `annotations_metadata.csv` file inside this repository's `data` directory. Rename `all_files` to `stormfront`, and `annotations_metadata.csv` to `stormfront.csv`.
