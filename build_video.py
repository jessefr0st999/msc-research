'''
Script for compiling images into a video using OpenCV-python. This can be useful
for animating forecasts, among other things.

The user must specify the folder in which the images are stored, and an
appropriate name for the resulting mp4 file.
'''
import argparse
import cv2
import os

def main():
    parser = argparse.ArgumentParser()
    # Folder where the images are stored and the video is created
    parser.add_argument('--target_folder', default='images')
    parser.add_argument('--video_name', default='video')
    args = parser.parse_args()

    # Make sorted listed of images, assuming all end in png
    images = [img for img in os.listdir(args.target_folder) if img.endswith(".png")]
    images.sort()
    # Get size of first image for reference
    frame = cv2.imread(os.path.join(args.target_folder, images[0]))
    height, width, layers = frame.shape
    # Define FourCC (4-byte video codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{args.target_folder}/{args.video_name}.mp4', fourcc, 3, (width,height))
    # Loop through the images and add to video
    for image in images:
        video.write(cv2.imread(os.path.join(args.target_folder, image)))
    # Tidy up
    cv2.destroyAllWindows()
    video.release()
    print(f'Saved to file {args.target_folder}/{args.video_name}.mp4!')

main()
