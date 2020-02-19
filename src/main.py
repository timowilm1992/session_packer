import gzip
import json
import os
import shutil
import logging
from tensorflow.python.lib.io.tf_record import TFRecordWriter, TFRecordCompressionType
from tensorflow.python.lib.io.tf_record import TFRecordOptions
from src.records import to_sequence_example
from src.transform import transform_example, drop_images, schema

FILESIZE = 500

cwd = '/'.join(os.getcwd().split('/')[:-1])
data_dir = f'{cwd}/data'
session_packer_version = 1
dummi_mnist_version = 1
session_packer_prefix = 'session-packer-' + str(session_packer_version).zfill(5)
dummi_mnist_prefix = 'dummi-mnist-' + str(dummi_mnist_version).zfill(5)
data_path = f'{data_dir}/{dummi_mnist_prefix}'
session_packer_path = f'{data_path}/{session_packer_prefix}'

def get_part_filename(old_version):
    return 'part-' + str(old_version).zfill(5)

def create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else: os.makedirs(path)


def write_success_file(out_path):
    with open(f'{out_path}/_SUCCESS','w'):
        pass

def write_schema(schema, path, schema_type='context_schema'):
    with open(f'{path}/{schema_type}', 'w') as file:
        file.write(json.dumps(schema)+'\n')

def write(data_path, session_packer_path, type='train'):

    out_path = f'{session_packer_path}/{type}'
    with gzip.open(f'{data_path}/{type}.gz', 'r') as file:
        partfile_number = 0

        for i, ex_str in enumerate(file):

            if (i + 1) % (FILESIZE + 1) == 0:
                partfile_number += 1

            part = get_part_filename(partfile_number)
            with TFRecordWriter(f'{out_path}/{part}', TFRecordOptions(compression_type=TFRecordCompressionType.GZIP)) as writer:
                example = json.loads(ex_str.decode())
                example = transform_example(example)
                drop_images(transformed_example=example)
                ctx_schema, seq_schema = schema(example)
                seq_example = to_sequence_example(ctx_schema, seq_schema, example)
                writer.write(seq_example.SerializeToString())
    write_schema(ctx_schema, out_path, 'context_schema')
    write_schema(seq_schema, out_path, 'sequence_schema')
    write_success_file(out_path)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info('Write train files.')
    train_out_path = f'{session_packer_path}/train'
    create_directory(train_out_path)
    write(data_path, session_packer_path, 'train')


    logger.info('Write test files.')
    test_out_path = f'{session_packer_path}/test'
    create_directory(test_out_path)
    write(data_path, session_packer_path, 'test')