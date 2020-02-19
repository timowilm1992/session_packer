from functools import partial

from tensorboard.compat.tensorflow_stub.dtypes import int64, float32
from tensorflow import FixedLenFeature
from tensorflow import FixedLenSequenceFeature
from tensorflow.core.example.example_pb2 import SequenceExample
from tensorflow.core.example.feature_pb2 import FeatureLists, Feature, FeatureList, Int64List, FloatList


def _float_feature(value):
    return Feature(float_list=FloatList(value=value))

def _int_feature(value):
    return Feature(int64_list=Int64List(value=value))

def _to_feature_list(value, trans_fn):
    return FeatureList(feature=[trans_fn(v) for v in value])


to_tf_feature_type = {'int64': partial(_int_feature), 'float64': partial(_float_feature)}

schema_type = {'int64': int64, 'float64': float32}




def build_schema(context_schema, sequence_schema):
    tf_context_schema = {feature: FixedLenFeature(shape=attributes['shape'], dtype=schema_type[attributes['type']]) for feature, attributes in context_schema.items()}
    tf_seq_schema = {feature: FixedLenSequenceFeature(shape=attributes['shape'], dtype=schema_type[attributes['type']]) for feature, attributes in sequence_schema.items()}
    return tf_context_schema, tf_seq_schema


def to_sequence_example(context_schema, seq_schema, example):
    context = {}
    feature_lists = {}
    for feature,value in example['context'].items():
        context[feature] = to_tf_feature_type[context_schema[feature]['type']](value)
    for feature, value_list in example['detailed_reco'].items():
        feature_lists[feature] = _to_feature_list(value_list, to_tf_feature_type[seq_schema[feature]['type']])
    return SequenceExample(context={'feature': context}, feature_lists=FeatureLists(feature_list=feature_lists))