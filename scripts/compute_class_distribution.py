import tensorflow as tf
import numpy as np

from collections import defaultdict
from absl import app, flags
from tqdm import tqdm
from libml import utils

flags.DEFINE_integer('seed', 0, 'Random seed to use, 0 for no shuffling.')
FLAGS = flags.FLAGS


def get_class(serialized_example):
    return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']


def main(argv):
    input_files = argv[1:]
    count = 0
    id_class = []
    class_id = defaultdict(list)

    print('Computing class distribution')
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(get_class, 4).batch(1 << 10)
    it = dataset.make_one_shot_iterator().get_next()
    try:
        with tf.Session() as session, tqdm(leave=False) as t:
            while 1:
                old_count = count
                for i in session.run(it):
                    id_class.append(i)
                    class_id[i].append(count)
                    count += 1
                t.update(count - old_count)
    except tf.errors.OutOfRangeError:
        pass
    print('%d records found' % count)
    nclass = len(class_id)
    train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64)
    train_stats /= count
    if 'stl10' in argv[1]:
        # All of the unlabeled data is given label 0, but we know that
        # STL has equally distributed data among the 10 classes.
        train_stats[:] *= 0
        train_stats[:] += 1

    print(
        "Frequencies : \n{}".format(
            ',\n'.join(["{} : {} %".format(i, 100 * train_stats[i]) for i in range(nclass)])
        )
    )
    # assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    # class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]
    # if FLAGS.seed:
    #     np.random.seed(FLAGS.seed)
    #     for i in range(nclass):
    #         np.random.shuffle(class_id[i])


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
