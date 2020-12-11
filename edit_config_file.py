import tensorflow as tf
from google.protobuf import text_format

from object_detection.protos import pipeline_pb2
from object_detection.protos.input_reader_pb2 import TFRecordInputReader

def get_configs_from_pipeline_file(pipeline_config_path, config_override=None):

  '''
  read .config and convert it to proto_buffer_object
  '''

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
  if config_override:
    text_format.Merge(config_override, pipeline_config)
  #print(pipeline_config)
  return pipeline_config

def write_configs_to_new_file(configs, new_file):
    with tf.io.gfile.GFile(new_file, "w") as f:
        f.write(configs)
    f.close()



configs = get_configs_from_pipeline_file('/home/lukas/coding/labelImg/pipeline copy.config')
# config_as_dict = create_configs_from_pipeline_proto(configs)

#use assignment to set values in configs
configs.model.ssd.num_classes = 4
config_text = text_format.MessageToString(configs)
# write_configs_to_new_file(config_text, '/home/lukas/coding/labelImg/pipeline new.config')

#use append to add new line to configs
configs.train_input_reader.tf_record_input_reader.input_path.append('test')
print(configs.train_input_reader.tf_record_input_reader)