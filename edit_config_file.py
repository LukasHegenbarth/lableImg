import tensorflow as tf
from google.protobuf import text_format

from object_detection.protos import pipeline_pb2

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


# def create_configs_from_pipeline_proto(pipeline_config):
#   '''
#   Returns the configurations as dictionary
#   '''

#   configs = {}
#   configs["model"] = pipeline_config.model
#   configs["train_config"] = pipeline_config.train_config
#   configs["train_input_config"] = pipeline_config.train_input_reader
#   configs["eval_config"] = pipeline_config.eval_config
#   configs["eval_input_configs"] = pipeline_config.eval_input_reader
#   # Keeps eval_input_config only for backwards compatibility. All clients should
#   # read eval_input_configs instead.
#   if configs["eval_input_configs"]:
#     configs["eval_input_config"] = configs["eval_input_configs"][0]
#   if pipeline_config.HasField("graph_rewriter"):
#     configs["graph_rewriter_config"] = pipeline_config.graph_rewriter

#   return configs

def write_configs_to_new_file(configs, new_file):
    with tf.io.gfile.GFile(new_file, "w") as f:
        f.write(configs)
    f.close()



configs = get_configs_from_pipeline_file('/home/lukas/coding/labelImg/pipeline copy.config')
# config_as_dict = create_configs_from_pipeline_proto(configs)
configs.model.ssd.num_classes = 4
config_text = text_format.MessageToString(configs)
write_configs_to_new_file(config_text, '/home/lukas/coding/labelImg/pipeline new.config')