import os
from PIL import Image
from argparse import ArgumentParser


def init_argparse():
    """
    Initialize argparse

    @return: parsed data in dictionary
    """
    parser = ArgumentParser(description='Database resizer')
    parser.add_argument(
        '-db',
        '--database',
        nargs='?',
        help='path to database',
        type=str)
    return parser

def resize_database(database_path):
    """
    Resize each image of database

    @param database_path: path to database, which content will be changed
    """
    for root, _, files in os.walk(database_path):
        for file in files:
            if file.endswith('.png'):
                img = Image.open(os.path.join(root, file))
                # resizing of each image in database
                width, height = img.size
                if (width < height):
                   coef = float(height/width)
                   new_width, new_height = 256, int(256*coef)
                else:
                   coef = float(width/height)
                   new_width, new_height = int(256*coef), 256
                img = img.resize((new_width, new_height), Image.ANTIALIAS)
                img.save(os.path.join(root, file)) # save resized image

def main():
    args = init_argparse().parse_args()
    database_path = args.database
    resize_database(database_path)


if __name__ == '__main__':
    main()