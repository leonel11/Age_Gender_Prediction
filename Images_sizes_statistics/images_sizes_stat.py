from glob import glob
from argparse import ArgumentParser
from PIL import Image


def init_argparse():
    """
    Initialize argparse

    @return: parsed data in dictionary
    """
    parser = ArgumentParser(description='SSD Statistics of sizes for test images')
    parser.add_argument(
        '-d',
        '--test_directory',
        nargs='?',
        help='path to test images',
        type=str)
    return parser


def build_sizes_stat(test_dir):
    """
    Build dictionary with statistics sizes of test images

    @param test_dir: directory with test images
    @return: dictionary with sizes of test images
    """
    stat_sizes = dict()
    images_lst = glob.glob(test_dir + '/*.jpg') # search by mask of files
    for img_file in images_lst:
        img = Image.open(img_file)
        width, height = img.size
        img_name = (img_file.split('/')[-1]).split('.')[0] # name of test image without extension
        stat_sizes[img_name] = [width, height]
    return stat_sizes


def output_sizes_stat(sizes_dict):
    """
    Output dictionary to file 'test_name_size.txt' for testing SSD

    @param sizes_dict: dictionary with sizes of test images
    """
    with open('test_name_size.txt', 'w') as f:
        for key in sizes_dict:
            f.write('{} {} {}\n'.format(key, sizes_dict[key][0], sizes_dict[key][1]))


def main():
    args = init_argparse().parse_args()
    test_dir = args.test_directory
    stat_sizes = build_sizes_stat(test_dir)
    output_sizes_stat(stat_sizes)


if __name__ == '__main__':
    main()