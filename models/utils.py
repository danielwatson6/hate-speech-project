"""Miscellaneous functions used exclusively by models."""


def infinite(dataset):
    """Create a copy of a `tf.data.Dataset` for endless validation.

    This allows to call the `next` to get a fresh validation batch.

    """
    return iter(dataset.repeat())


def initialize_embeds(embeds_layer, embedding_matrix):
    """Sets the weights of a `tf.keras.layers.Embedding` layer to the given matrix."""
    if len(embeds_layer.weights) == 0:
        raise RuntimeError("Embedding layer hasn't been built.")

    embeds_layer.weights[0].assign(embedding_matrix)


def make_word_id_maps(vocab_path, vocab_size):
    """Build word-to-id and id-to-word functions from a TSV file."""

    # Args: filename, key_dtype, key_index, value_dtype, value_index, vocab_size.
    word_to_id_init = tf.lookup.TextFileInitializer(
        vocab_path,
        tf.string,
        0,
        tf.int64,
        tf.lookup.TextFileIndex.LINE_NUMBER,
        vocab_size=vocab_size,
    )
    id_to_word_init = tf.lookup.TextFileInitializer(
        vocab_path,
        tf.int64,
        tf.lookup.TextFileIndex.LINE_NUMBER,
        tf.string,
        0,
        vocab_size=vocab_size,
    )
    word_to_id = tf.lookup.StaticHashTable(word_to_id_init, 1)
    id_to_word = tf.lookup.StaticHashTable(id_to_word_init, "<unk>")

    return word_to_id.lookup, id_to_word.lookup
