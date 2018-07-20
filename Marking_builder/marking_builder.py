import os
from argparse import ArgumentParser

def init_argparse():
    """
    Initialize argparse

    @return: parsed data in dictionary
    """
    parser = ArgumentParser(description='Marking builder')
    parser.add_argument(
        '-s',
        '--selection',
        nargs='?',
        help='path to database, where data are marked by folders',
        type=str)
    return parser

def get_images(database_path):
    """
    Get all images of selection

    @param database_path: path to selection
    @return: list of images
    """
    images = []
    for root, _, files in os.walk(database_path):
        for file in files:
            if file.endswith('.png'):
                images.append(os.path.join(root, file))
    return images

def get_labels(images):
    """
    Get all labels of selection

    @param images: list of images in selection
    @return: list of labels
    """
    labels = []
    for idx, rec in enumerate(images):
        pos = rec.find('\\')
        images[idx] = rec[pos + 1:].replace('\\', '/')
        if 'F/' in images[idx]:
            labels.append(0) # female
        else:
            labels.append(1) # male
    return labels

def build_marking(database_path):
    """
    Build marked file

    @param database_path: path to selection
    """
    images = get_images(database_path)
    labels = get_labels(images)
    selection_file = os.path.join(database_path, 'selection.txt')
    with open(selection_file, 'w') as f:
        for i in range(len(images)):
            f.write(images[i] + ' ' + str(labels[i]) + '\n')

def main():
    args = init_argparse().parse_args()
    selection = args.selection
    build_marking(selection)

if __name__ == '__main__':
    main()