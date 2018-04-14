import tensorflow as tf
from tensor2tensor.utils import registry, modality

__all__ = [
    "RelativeTreeDistanceModality",
]

#
# def _parse_tree_string(x):
#     ret = []
#     for lev_id, level in enumerate(x.split('-')):
#         for node in level.split(','):
#             node_id = int(node)
#             while node_id <= len(ret):
#                 ret.append(0)
#             ret[node_id] = lev_id + 1
#     return ret


@registry.register_symbol_modality("relative_tree_distance")
class RelativeTreeDistanceModality(modality.Modality):
    def bottom(self, x):
        max_length = tf.shape(x)[-1]
        t = tf.map_fn(
            lambda y: tf.sparse_tensor_to_dense(
                # dirty fix: max_length+1
                tf.sparse_reset_shape(tf.string_split(y, ','), new_shape=[max_length+1, max_length+1]),
                default_value="0"),
            x)
        return tf.string_to_number(t)
