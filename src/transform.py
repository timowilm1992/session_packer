import numpy as np

def transform_detailed_reco(detailed_reco):
    d_rec = {}
    for reco in detailed_reco:
        for k, v in reco.items():
            d_rec[k] = d_rec.get(k, [])
            d_rec[k].append(v) if k != 'lbl_key' else d_rec[k].append([v])
    return d_rec


def drop_images(transformed_example):
    transformed_example['context'].pop('anchor_image')
    transformed_example['detailed_reco'].pop('image')


def transform_example(example):
    return {'context': {'anchor': [example['anchor']],
                        'anchor_image': example['anchor_image'],
                        'anchor_label': example['anchor_label'],
                        'anchor_lbl_key': [example['anchor_lbl_key']],
                        'context': [example['context']],
                        'context_vec': example['context_vec'],
                        'click_position': [example['click_position']],
                        'reco': example['reco'],
                        'seen_click_position': [example['seen_click_position']],
                        'seen_mask': example['seen_mask']},
            'detailed_reco': transform_detailed_reco(example['detailed_reco'])}


def schema(transformed_example):
    context_schema = {k: {'type': str(np.array(v).dtype) , 'shape': [int(d) for d in np.shape(v)]} for k,v in transformed_example['context'].items()}
    seq_schema = {k: {'type': str(np.array(v).dtype) , 'shape': [int(d) for d in np.shape(v)][1:]} for k,v in transformed_example['detailed_reco'].items()}
    return context_schema, seq_schema
