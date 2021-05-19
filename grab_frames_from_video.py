import cv2 
import argparse
import os
from pathlib import Path

# TODO make user input: sample rate in images per second
# TODO get frame rate from video to calculate framerate

def arg_parser():
    desc = "divide video in frames"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-v', '--video', type=str, default=None, required=True)
    parser.add_argument('-r', '--imgs_per_second', type=float, default=1.0, required=False)
    return check_args(parser.parse_args())

def check_args(args):
    check_path(args.video)
    return args


def check_path(path):
    if not os.path.exists(path):
        print('PathError: The specified folder {} does not exist'.format(path))
    return Path(path)


if __name__ == "__main__":
    args = arg_parser()
    outdir = args.video.split('.')[0]
    os.mkdir(outdir)

    rate = int(30/args.imgs_per_second)
    print("saving {} frames per second".format(args.imgs_per_second))
    cap = cv2.VideoCapture(args.video)
    frame_id = 0

    while cap.isOpened():
        ok, frame = cap.read()
        
        if not ok:
            print("Can't process frame. Exiting...")
            break
        
        #save frame every 10 seconds
        if frame_id % rate == 0:
            cv2.imwrite(outdir + '/' + str(frame_id//rate) + '.jpg', frame)


        frame_id += 1
