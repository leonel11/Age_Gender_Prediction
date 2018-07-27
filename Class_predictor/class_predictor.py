import sys
import os
import numpy as np
import caffe
import matplotlib.pyplot as plt
import pandas as pd

def getCountClases(mode):
    if mode == 1: # gender_prediction mode
        return 2
    if mode == 2: # age prediction
        return 6
    return 0


def getClassNumber(prototxt_file, caffemodel_file, picture_file):
    image = caffe.io.load_image(picture_file, color=False)
    # apply transformer in preprocessing of image
    transformed_image = transformer.preprocess('data', image)
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image
    # perform classification
    output = net.forward(end='loss')
    output_prob = output['loss'] # the output probability vector for the image
    predicted_class = output_prob.argmax()
    return predicted_class

def calcScore(count_classes, selection_path, selection_file, prototxt_file, caffemodel_file):
    data = pd.DataFrame(columns=['filename', 'real', 'predicted']) # table of images and classes (true and predicted)
    f = open(selection_file, 'r')
    # form table
    for line in f:
        str_subs = line.split()
        filename = str_subs[0]
        real_class = str_subs[1]
        predicted_class = getClassNumber(prototxt_file, caffemodel_file, os.path.join(selection_path, filename))
        data.loc[len(data)] = [filename, real_class, predicted_class]
    data.to_csv('train_res.csv', sep = ' ') # record classification data

if len(sys.argv) != 6: # help, how to run script
    print('Script must be run in the format:\n python ConfusionMatrix.py mode selection_path selection_file deploy_prototxt caffemodel_file\n')
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
    count_classes = getCountClases(mode)
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
    calcScore(count_classes, selection_path, selection_file, prototxt_file, caffemodel_file)