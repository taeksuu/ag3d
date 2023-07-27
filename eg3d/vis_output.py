import os
import cv2
import argparse
import glob
import ffmpeg
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_dir', type=str, required=True)
args = parser.parse_args()

output_dir = args.file_dir

raw_imgs = sorted(glob.glob(f'{output_dir}/fakes*_raw.png'))

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (205,246)
topText = (110,20)
fontScale              = 0.5
fontColor              = (0,0,0)
thickness              = 1
lineType               = 2

os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
for i, img in enumerate(raw_imgs):
    # shutil.copy(img, os.path.join(output_dir, 'raw', f'{i:05d}.png'))
    big = cv2.imread(img)
    big = big[:256*1, :256*1]
    cv2.putText(big,f'{i}K', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    cv2.putText(big,'EG3D', 
        topText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    cv2.imwrite(os.path.join(output_dir, 'raw', f'{i:05d}.png'), big)

cmd = 'ffmpeg -framerate 40 -i ' + output_dir + '/raw/%05d.png -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + os.path.join(output_dir + 'raw.mp4')
os.system(cmd)

raw_imgs = [img.replace('_raw', '_depth') for img in raw_imgs]
os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
for i, img in enumerate(raw_imgs):
    # shutil.copy(img, os.path.join(output_dir, 'depth', f'{i:05d}.png'))
    big = cv2.imread(img)
    big = big[:256*1, :256*1]
    cv2.putText(big,f'{i}K', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    cv2.putText(big,'EG3D', 
        topText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    cv2.imwrite(os.path.join(output_dir, 'depth', f'{i:05d}.png'), big)

cmd = 'ffmpeg -framerate 40 -i ' + output_dir + '/depth/%05d.png -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + os.path.join(output_dir + 'depth.mp4')
os.system(cmd)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (450*2,492*2)
topText = (235*2,40)
fontScale              = 1
fontColor              = (0,0,0)
thickness              = 2
lineType               = 2


raw_imgs = [img.replace('_depth', '') for img in raw_imgs]
os.makedirs(os.path.join(output_dir, 'out'), exist_ok=True)
for i, img in enumerate(raw_imgs):
    # shutil.copy(img, os.path.join(output_dir, 'out', f'{i:05d}.png'))
    big = cv2.imread(img)
    big = big[:256*4, :256*4]
    cv2.putText(big,f'{i}K', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    cv2.putText(big,'EG3D', 
        topText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    cv2.imwrite(os.path.join(output_dir, 'out', f'{i:05d}.png'), big)

cmd = 'ffmpeg -framerate 40 -i ' + output_dir + '/out/%05d.png -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + os.path.join(output_dir + 'out.mp4')
os.system(cmd)