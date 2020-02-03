# Developing data loaders

Textual input pipelines in TensorFlow can be a bit tricky, so this README should serve as a guide on best practices on how to develop these.

General premises:

### 1. Data-cleaning within the data loader should be as minimal as possible for maintainability, but also to make the input pipeline itself faster.

As much as the data cleaning as possible should happen in its own script that developers can quickly run to clean the input data. Exceptions where the data loader should do more could be things like, "is using punctuation helpful or hurtful to the model? lowercasing everything vs. not?", which can be passed as hyperparameters.

The cleaning script should produce a clean data file(s), but also a sorted vocabulary TSV (explained below). The first few "words" aren't actually words though-- they're reserved special id's. The first id, 0, is ALWAYS reserved for padding, which adds trailing 0's to fit sequences of different length into a tensor. E.g.,
```
["hello world", "hello im daniel watson"]

gets mapped to something like this (note the 0's):

[[27 101 0 0], [27 1 202 2921]]

```

The second id, 1, should always be reserved for out-of-vocabulary, unknown words. Whenever encountering a token not included in the vocabulary, that id is used.

Seq2Seq models usually require a "start of sentence" token, and both seq2seq and language models require an "end of sentence" token to ignore everything outputted afterwards, so by convention, those should be included too.

The TSV file should thus look like this (`\t` indicates a tab):

```
<pad>\t0
<unk>\t[number_oov_tokens]
<sos>\t0
<eos>\t0
the\t1201231
and\t615212
of\t318219
...
```

### 2. Models that work with textual data should be developed in a way that is data-agnostic.

To achieve this, the data loader should be the entity that handles string <-> index conversion (i.e., when every word is converted to a corresponding index or vice versa).

This project has the module `data_loaders/utils.py` to handle a lot of this. For instance, `utils.make_word_id_maps` takes a TSV file where all the lines have format `[word]\t[count]` and that is sorted by descending word frequencies **in the training data**. The typical pattern is this:
```python
self.word_to_id, self.id_to_word = utils.make_word_id_maps(
  path_to_tsv_file, self.hparams.vocab_size
)
```

The model can then use `data_loader.id_to_word(output_ids)`, for example, to display outputs as strings.

Another challenge is how to make the embedding matrix in the model agnostic to the data loader (because it depends on the vocabulary, which itself depends on the training data). `data_loaders/utils.py` has a solution for this too, which will use Google word2vec embeddings for the words in the vocabulary:
```python
self.embedding_matrix = utils.save_or_load_embeds(
   path_to_npy_file, path_to_tsv_file, self.hparams.vocab_size
)
```

where `path_to_npy_file` is a string like `"data/my_dataset/something.npy"` that this method will create if necessary. This is a file containing the numpy array for the embeddings. The model can use this as follows:
```python
# At the top
from models import utils

# Inside `fit` method
if self.step.numpy() == 0:
    utils.initialize_embeds(self.embeds, data_loader.embedding_matrix)
```

Where `self.embeds` is a `tf.keras.layers.Embedding` instance  and `self.step` is a TensorFlow variable to keep track of training steps (allows to remember where training was stopped if it was interrupted and later resumed).

### 3. The data loader should convert words to integers

This avoids the model being cluttered with low-level logic to handle this, further textual preprocessing, padding, etc.

The typical pattern after instantiating the dataset instance(s):
```python
# This is inside `call()` or a method used in `call()`
# `dataset` is a `tf.data.Dataset` instance
dataset = dataset.shuffle(10000)  # only for training and validation, and must happen BEFORE batching to get truly random shuffle
dataset = dataset.batch(self.hparams.batch_size)
dataset = dataset.map(self._batch_to_ids)
dataset = dataset.prefetch(1)  # this must go at the very end

# This is `_batch_to_ids()`
def _batch_to_ids(self, batch):
    # NOTE: this returns a RaggedTensor!
    # https://www.tensorflow.org/guide/ragged_tensor
    sequences = tf.strings.split(batch)
    # Further textual preprocessing happens here, before converting `sequences` to a regular tensor.
    padded = sequences.to_tensor(default_value="<pad>")
    if self.hparams.max_seq_len:
        padded = padded[:, :self.hparams.max_seq_len]
    return self.word_to_id(padded)
```
