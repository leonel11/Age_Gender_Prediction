import sys
import os
import numpy as np
import caffe
import matplotlib.pyplot as plt
import pandas as pd


def get_count_clases(mode):
    '''
    Get amount of clases according to changed mode

    @param mode: changed mode
    @return: amount of classes
    '''
    if mode == 1: # gender_prediction mode
        return 2
    if mode == 2: # age_prediction
        return 6
    return 0


def get_class_number(prototxt_file, caffemodel_file, picture_file):
    '''
    Get number of predicted class for picture

    @param prototxt_file: file with the architecture of CNN
    @param caffemodel_file: weights of CNN
    @param picture_file: file for classification
    @return: predicted class
    '''
    image = caffe.io.load_image(picture_file, color=False)
    # apply transformer in preprocessing of image
    transformed_image = transformer.preprocess('data', image)
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image
    # perform classification
    output = net.forward(end='loss')
    output_prob = output['loss'] # the output probability vector for the image
    predicted_class = output_prob.argmax() # choose predicted class
    return predicted_class


def build_classification_table(count_classes, selection_path, selection_file, prototxt_file, caffemodel_file):
    '''
    Build a table with classification results

    @param count_classes: number of classes (2 - for gender_prediction? 6 - for age prediction)
    @param selection_path: path to the folder with pictures
    @param selection_file: file with list of all images for classification
    @param prototxt_file: file with the architecture of CNN
    @param caffemodel_file: weights of CNN
    '''
    data = pd.DataFrame(columns=['filename', 'real', 'predicted']) # table of images and classes (true and predicted)
    f = open(selection_file, 'r')
    # form table of classification
    for line in f:
        str_subs = line.split()
        filename = str_subs[0]
        real_class = str_subs[1]
        predicted_class = get_class_number(prototxt_file, caffemodel_file, os.path.join(selection_path, filename))
        data.loc[len(data)] = [filename, real_class, predicted_class]
    f.close()
    data.to_csv('train_res.csv', sep = ' ') # record classification data


def main():
    if len(sys.argv) != 6: # help, how to run script
        print('Script must be run in the format:\n '
            'python ConfusionMatrix.py mode selection_path selection_file deploy_prototxt caffemodel_file\n')
        print('Modes of script:')
        print('     1 - for gender prediction')
        print('     2 - for age prediction')
    else:
    	# reading params of script
        mode = int(sys.argv[1])
        selection_path = str(sys.argv[2])
        selection_file = str(sys.argv[3])
        prototxt_file = str(sys.argv[4])
        caffemodel_file = str(sys.argv[5])
        count_classes = get_count_clases(mode)
        # Caffe tuning
        caffe.set_device(0)
        caffe.set_mode_gpu()
        net = caffe.Net(prototxt_file,  # deploy prototxt model
                        caffemodel_file,  # trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)
        #  create transformer for the input image from 'data' layer
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        transformer.set_mean('data', np.array([127.0, 127.0, 127.0]))
        # transformer.set_raw_scale('data', 255.0)
        net.blobs['data'].reshape(1,  # batch size
                                  3,  # 3-channel (BGR) images
                                  60, 60) # image size
        build_classification_table(count_classes, selection_path, selection_file, prototxt_file, caffemodel_file)


if __name__ == '__main__':
    main()