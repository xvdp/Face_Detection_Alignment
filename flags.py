from typing import Any, List, Tuple, Union
import tensorflow as tf

from distutils.version import LooseVersion

class EasyDict(dict):
    """Replace FLAGS with Easy Dict, fails under tf > 1.12 when retrieving values
        code borrowed from stylegan2
    """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

if LooseVersion(tf.__version__) >= LooseVersion("1.12"):
    _tf = tf.compat.v1
else:
    _tf = tf.app


_FLAGS = _tf.flags.FLAGS

#_tf.flags.DEFINE_float(<name>, <val>, <docstring>)
_tf.flags.DEFINE_float('initial_learning_rate', 0.0001, '''Initial learning rate.''')
_tf.flags.DEFINE_float('num_epochs_per_decay', 5.0, '''Epochs after which learning rate decays.''')
_tf.flags.DEFINE_float('learning_rate_decay_factor', 0.97, '''Learning rate decay factor.''')
_tf.flags.DEFINE_float('learning_rate_decay_step', 30000,'''Learning rate decay factor.''')

_tf.flags.DEFINE_integer('batch_size', 4, '''The batch size to use.''')
_tf.flags.DEFINE_integer('eval_size', 4, '''The batch size to use.''')
_tf.flags.DEFINE_integer('num_iterations', 2, '''The number of iterations to unfold the pose machine.''')
_tf.flags.DEFINE_integer('num_preprocess_threads', 4,'''How many preprocess threads to use.''')
_tf.flags.DEFINE_integer('n_landmarks', 84,'''number of landmarks.''')
_tf.flags.DEFINE_integer('rescale', 256,'''Image scale.''')

_tf.flags.DEFINE_string('dataset_dir', '', '''Directory where to load datas.''')
_tf.flags.DEFINE_string('train_dir', 'ckpt/train', '''Directory where to write event logs and checkpoint.''')
_tf.flags.DEFINE_string('eval_dir', '','''Directory where to write event logs and checkpoint.''')
_tf.flags.DEFINE_string('graph_dir', 'model/weight.pkl','''If specified, restore this pretrained model.''')

_tf.flags.DEFINE_integer('max_steps', 1000000,'''Number of batches to run.''')
_tf.flags.DEFINE_string('train_device', '/gpu:0','''Device to train with.''')

_tf.flags.DEFINE_integer('flip_pred', 0,'''db name.''')

_tf.flags.DEFINE_string('train_model', '', '''training model.''')
_tf.flags.DEFINE_string('pretrained_model_checkpoint_path', '', '''Restore pretrained model.''')
_tf.flags.DEFINE_string('testset_name', '', '''test set name.''')
_tf.flags.DEFINE_string('model_name', '', '''test model name.''')
_tf.flags.DEFINE_string('savemat_name', '', '''save_mat_name''')

if LooseVersion(tf.__version__) >= LooseVersion("1.12"):
    FLAGS = EasyDict()
    for key in _FLAGS._flags():
        FLAGS[key] = _FLAGS._flags()[key].value
else:
    FLAGS = _FLAGS