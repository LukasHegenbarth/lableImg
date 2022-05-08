from create_pascal_tf_record import convert_folder_to_tfrecord

convert_folder_to_tfrecord(['/home/lukas/Videos/sugarbeet_2021/23_05_2021/2021-05-14_07-40-25_camera_1', '/home/lukas/Videos/sugarbeet_2021/30_05_2021/2021-05-14_08-17-00_camera_1', '/home/lukas/Videos/sugarbeet_2021/30_05_2021/2021-05-14_08-24-58_camera_1' ], 'train', '/home/lukas/training_workspace/data/beet/beet_label_map.pbtxt', '/home/lukas/training_workspace/data/beet/TFRecords/eval.record')
