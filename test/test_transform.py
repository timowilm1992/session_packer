from unittest import TestCase

from src.transform import transform_detailed_reco, transform_example, schema, drop_images


class test_reader(TestCase):
    def test_transform_detailed_reco(self):
        detailed_reco = [{'image': [3., 3., 3.], 'label': [0., 0., 1., 0.], 'lbl_key': 2, 'context': [1., 0., 0.]},
                         {'image': [2., 2., 2.], 'label': [0., 1., 0., 0.], 'lbl_key': 1, 'context': [0., 1., 0.]},
                         {'image': [1., 1., 1.], 'label': [1., 0., 0., 0.], 'lbl_key': 0, 'context': [0., 0., 1.]}]

        transformed_detailed_reco = {'image': [[3., 3., 3.], [2., 2., 2.], [1., 1., 1.]],
                                     'label': [[0., 0., 1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.]],
                                     'lbl_key': [[2], [1], [0]],
                                     'context': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}

        self.assertDictEqual(transform_detailed_reco(detailed_reco), transformed_detailed_reco)

    def test_transform_example(self):
        example = {'anchor': 0,
                   'anchor_image': [1., 1., 1.],
                   'anchor_label': [1., 0., 0., 0.],
                   'anchor_lbl_key': 0,
                   'context': 1,
                   'context_vec': [0., 1.],
                   'click_position': 2,
                   'reco': [2, 1, 0],
                   'seen_click_position': 0,
                   'seen_mask': [1., 0., 1.],
                   'detailed_reco': [
                       {'image': [3., 3., 3.], 'label': [0., 0., 1., 0.], 'lbl_key': 2, 'context': [1., 0., 0.]},
                       {'image': [2., 2., 2.], 'label': [0., 1., 0., 0.], 'lbl_key': 1, 'context': [0., 1., 0.]},
                       {'image': [1., 1., 1.], 'label': [1., 0., 0., 0.], 'lbl_key': 0, 'context': [0., 0., 1.]}]}

        transformed_example = {'context': {'anchor': [0],
                                           'anchor_image': [1., 1., 1.],
                                           'anchor_label': [1., 0., 0., 0.],
                                           'anchor_lbl_key': [0],
                                           'context': [1],
                                           'context_vec': [0., 1.],
                                           'click_position': [2],
                                           'reco': [2, 1, 0],
                                           'seen_click_position': [0],
                                           'seen_mask': [1., 0., 1.]},
                               'detailed_reco':
                                   {'image': [[3., 3., 3.], [2., 2., 2.], [1., 1., 1.]],
                                    'label': [[0., 0., 1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.]],
                                    'lbl_key': [[2], [1], [0]],
                                    'context': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}}

        self.assertDictEqual(transform_example(example), transformed_example)


    def test_schema(self):
        transformed_example = {'context': {'anchor': [0],
                                           'anchor_image': [1., 1., 1.],
                                           'anchor_label': [1., 0., 0., 0.],
                                           'anchor_anchor_lbl_key': [0],
                                           'context': [1],
                                           'context_vec': [0., 1.],
                                           'click_position': [2],
                                           'reco': [2, 1, 0],
                                           'seen_click_position': [0],
                                           'seen_mask': [1., 0., 1.]},
                               'detailed_reco': {'image': [[3., 3., 3.], [2., 2., 2.], [1., 1., 1.]],
                                                 'label': [[0., 0., 1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.]],
                                                 'lbl_key': [[2], [1], [0]],
                                                 'context': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}}

        ctx_schema = {'anchor': {'type': 'int64', 'shape': [1]},
                      'anchor_image': {'type': 'float64', 'shape': [3]},
                      'anchor_label': {'type': 'float64', 'shape': [4]},
                      'anchor_anchor_lbl_key': {'type': 'int64', 'shape': [1]},
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

        context_schema, sequence_schema = schema(transformed_example)

        self.assertDictEqual(context_schema, ctx_schema)
        self.assertDictEqual(seq_schema, sequence_schema)

    def test_drop_images(self):
        transformed_example = {'context': {'anchor': [0],
                                           'anchor_image': [1., 1., 1.],
                                           'anchor_label': [1., 0., 0., 0.],
                                           'anchor_lbl_key': [0],
                                           'context': [1],
                                           'context_vec': [0., 1.],
                                           'click_position': [2],
                                           'reco': [2, 1, 0],
                                           'seen_click_position': [0],
                                           'seen_mask': [1., 0., 1.]},
                               'detailed_reco': {'image': [[3., 3., 3.], [2., 2., 2.], [1., 1., 1.]],
                                                 'label': [[0., 0., 1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.]],
                                                 'lbl_key': [[2], [1], [0]],
                                                 'context': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}}

        example_without_images = {'context': {'anchor': [0],
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

        drop_images(transformed_example)

        self.assertDictEqual(transformed_example, example_without_images)
