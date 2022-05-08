import cv2 
import argparse
import os
from pathlib import Path
import glob

# TODO make user input: sample rate in images per second
# TODO get frame rate from video to calculate framerate

def arg_parser():
    desc = "divide video in frames"
    parser = argparse.ArgumentParser(description=desc)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-v', '--video', type=str, default=None)
    group.add_argument('-f', '--folder', type=str, default=None)
    parser.add_argument('-r', '--imgs_per_second', type=float, default=1.0, required=False)
    return check_args(parser.parse_args())

def check_args(args):
    if args.video:
        check_path(args.video)
    if args.folder:
        check_path(args.folder)
    return args


def check_path(path):
    if not os.path.exists(path):
        print('PathError: The specified folder {} does not exist'.format(path))
    return Path(path)


if __name__ == "__main__":
    args = arg_parser()
    if args.video:
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
            filename = f'{outdir}/{frame_id:05d}.jpg'
            if frame_id % rate == 0:
                cv2.imwrite(filename, frame)
            
            frame_id += 1
    
    if args.folder:
        print(args.folder)
        videos = glob.glob(args.folder + '*camera_1.mkv')
        for video in videos:
            outdir = video.split('.')[0]
            if os.path.exists(outdir):
                print('Video has already been processed. Folder exists.')
            else:
                os.mkdir(outdir)

                rate = int(30/args.imgs_per_second)
                print("saving {} frames per second".format(args.imgs_per_second))

                cap = cv2.VideoCapture(video)
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
