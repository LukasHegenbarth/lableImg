from create_pascal_tf_record import convert_folder_to_tfrecord

convert_folder_to_tfrecord('/home/lukas/training_workspace/data/beet/video_frames/2021-05-14_08-08-09_camera_1', 'train', '/home/lukas/training_workspace/data/beet/beet_label_map.pbtxt', '/home/lukas/Videos/tfrecord/train_test.record')
