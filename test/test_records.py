import pathlib

from tensorboard.compat.tensorflow_stub.dtypes import int64, float32
from tensorflow import FixedLenFeature
from tensorflow import FixedLenSequenceFeature
from tensorflow import parse_single_sequence_example
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType, TFRecordOptions, TFRecordWriter, \
    tf_record_iterator
from tensorflow.python.framework.test_util import TensorFlowTestCase
from src.records import to_sequence_example, build_schema

test_dir = pathlib.Path(__file__).parent

class TestRecords(TensorFlowTestCase):
    def test_build_schema(self):
        ctx_schema = {'anchor': {'type': 'int64', 'shape': [1]},
                      'anchor_image': {'type': 'float64', 'shape': [3]},
                      'anchor_label': {'type': 'float64', 'shape': [4]},
                      'anchor_lbl_key': {'type': 'int64', 'shape': [1]},
                      'context': {'type': 'int64', 'shape': [1]},
                      'context_vec': {'type': 'float64', 'shape': [2]},
                      'click_position': {'type': 'int64', 'shape': [1]},
                      'reco': {'type': 'int64', 'shape': [3]},
                      'seen_click_position': {'type': 'int64', 'shape': [1]},
                      'seen_mask': {'type': 'float64', 'shape': [3]}}

        seq_schema = {'image': {'type': 'float64', 'shape': [3]},
                      'label': {'type': 'float64', 'shape': [4]},
                      'lbl_key': {'type': 'int64', 'shape': [1]},
                      'context': {'type': 'float64', 'shape': [3]}}

        expected_tf_ctx_schema = {'anchor': FixedLenFeature(shape=[1], dtype=int64),
                                  'anchor_image': FixedLenFeature(shape=[3], dtype=float32),
                                  'anchor_label': FixedLenFeature(shape=[4], dtype=float32),
                                  'anchor_lbl_key': FixedLenFeature(shape=[1], dtype=int64),
                                  'context': FixedLenFeature(shape=[1], dtype=int64),
                                  'context_vec': FixedLenFeature(shape=[2], dtype=float32),
                                  'click_position': FixedLenFeature(shape=[1], dtype=int64),
                                  'reco': FixedLenFeature(shape=[3], dtype=int64),
                                  'seen_click_position': FixedLenFeature(shape=[1], dtype=int64),
                                  'seen_mask': FixedLenFeature(shape=[3], dtype=float32)}

        expected_tf_seq_schema = {'image': FixedLenSequenceFeature(shape=[3], dtype=float32),
                                  'label': FixedLenSequenceFeature(shape=[4], dtype=float32),
                                  'lbl_key': FixedLenSequenceFeature(shape=[1], dtype=int64),
                                  'context': FixedLenSequenceFeature(shape=[3], dtype=float32)}

        tf_ctx_schema, tf_seq_schema = build_schema(ctx_schema, seq_schema)

        self.assertDictEqual(tf_ctx_schema, expected_tf_ctx_schema)
        self.assertDictEqual(tf_seq_schema, expected_tf_seq_schema)



    def test_to_sequence_example(self):
        example = {'context': {'anchor': [0],
                               'anchor_label': [1., 0., 0., 0.],
                               'anchor_lbl_key': [0],
                               'context': [1],
                               'context_vec': [0., 1.],
                               'click_position': [2],
                               'reco': [2, 1, 0],
                               'seen_click_position': [0],
                               'seen_mask': [1., 0., 1.]},
                   'detailed_reco': {'label': [[0., 0., 1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.]],
                                     'lbl_key': [[2], [1], [0]],
                                     'context': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}}


        ctx_schema = {'anchor': {'type': 'int64', 'shape': [1]},
                      'anchor_label': {'type': 'float64', 'shape': [4]},
                      'anchor_lbl_key': {'type': 'int64', 'shape': [1]},
                      'context': {'type': 'int64', 'shape': [1]},
                      'context_vec': {'type': 'float64', 'shape': [2]},
                      'click_position': {'type': 'int64', 'shape': [1]},
                      'reco': {'type': 'int64', 'shape': [3]},
                      'seen_click_position': {'type': 'int64', 'shape': [1]},
                      'seen_mask': {'type': 'float64', 'shape': [3]}}

        seq_schema = {'label': {'type': 'float64', 'shape': [4]},
                      'lbl_key': {'type': 'int64', 'shape': [1]},
                      'context': {'type': 'float64', 'shape': [3]}}

        tf_ctx_schema, tf_seq_schema = build_schema(ctx_schema, seq_schema)


        sequence_example = to_sequence_example(ctx_schema, seq_schema, example)

        tf_record_options = TFRecordOptions(TFRecordCompressionType.GZIP)
        part_filename =f'{test_dir}/resources/part-00001'

        with TFRecordWriter(part_filename, options=tf_record_options) as writer:
            writer.write(sequence_example.SerializeToString())

        record_iterator = tf_record_iterator(path=part_filename, options=tf_record_options)

        with self.test_session() as sess:
            context, sequences = parse_single_sequence_example(next(record_iterator), tf_ctx_schema, tf_seq_schema)
            ctx, sequences = sess.run([context, sequences])

            ctx = {k: v.tolist() for k,v in ctx.items()}
            sequences = {k: v.tolist() for k,v in sequences.items()}

            self.assertDictEqual(ctx, example['context'])

            self.assertDictEqual(sequences, example['detailed_reco'])

