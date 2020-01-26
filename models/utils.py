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
