import sys
import os
import numpy as np
import caffe

if len(sys.argv) != 4:
    print('Script must be run in the format:\n python output_ssd.py selection_path prototxt caffemodel\n')
else:
    selection_path = str(sys.argv[1])
    prototxt_file = str(sys.argv[2])
    caffemodel_file = str(sys.argv[3])
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(prototxt_file,  # deploy prototxt model
                    caffemodel_file,  # trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)
    #  create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    # transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    # transformer.set_mean('data', np.array([127]))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1,  # batch size
                              3,  # 3-channel (BGR) images
                              300, 300)
                              #122, 122)  # image size is 122x122
    list_pictures = os.listdir(selection_path)
    for pict in list_pictures:
    	print(pict)
        image = caffe.io.load_image(picture_file, color=True)
        transformed_image = transformer.preprocess('data', image)
        # copy the image data into the memory allocated for the net
        #mean_image = np.zeros([122, 122])
        mean_image = np.fill((300, 300), np.array([104.0, 117.0, 123.0]))
        net.blobs['data'].data[...] = transformed_image - mean_image
        # perform classification
        output = net.forward(end='detection_out')
        output_prob = output['detection_out'] # the output probability vector for the first image in the batch
        np.set_printoptions(formatter={'float': '{: .6f}'.format})
        print(output_prob)