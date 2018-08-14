'''
Batch run of script for the launch and visualization SSD
'''

import subprocess
import os


input_dir = "ssd_data/test_images/" # folder with images for work
files = os.listdir(input_dir) # list of images for work
output_dir = "ssd_data/res_ssd/" # folder for output results


for f in files:
    subprocess.call(['python', 'ssd_detectvis.py',
                     '--gpu_id=0',
                     '--labelmap_file=labelmap_gender.prototxt',
                     '--model_def=jobs/VGGNet/SSD_300x300/deploy.prototxt',
                     '--image_resize=300',
                     '--model_weights=snapshots/SSD_300x300/VGG_gender_SSD_300x300_iter_120000.caffemodel',
                     '--image_file={}{}'.format(input_dir, f),
                     '--output_image_file={}{}'.format(output_dir, f)])