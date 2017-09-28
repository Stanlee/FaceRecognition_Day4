"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import detect_face
import cv2
#from compare_temp_ex import face_alignment

def main(args):

    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO: create video capture class (cv2 package)
    # cap = cv2. video catpture
    cap = cv2.VideoCapture(0)
    #print(cap.get(3))
    #print(cap.get(4))

    # TODO: set the paramters about MT-CNN (minsize, threshold, scale factor)
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold, each network's threshold
    factor = 0.709  # scale factor pyramid

    # TODO: create MT-CNN network
    # create mtcnn
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction) # Amount of memory to use
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) # Using GPU options
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    with tf.Graph().as_default():
      
        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            # Load the model
            facenet.load_model(args.model)

            # TODO: load classifier model
            # pickle.load refered
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            with open(classifier_filename_exp, 'rb') as infile:  # load the model to predict the class
                (model, class_names) = pickle.load(infile)

            # TODO: get input and output tensors about embedding network
            # images_placeholder =
            # embeddings =
            # phase_train_placeholder =

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            #feed_dict = {images_placeholder: frame, phase_train_placeholder: False}
            #emb = sess.run(embeddings, feed_dict=feed_dict)

            while (cap.isOpened()):
                ret, frame = cap.read() # ret boolean
                if ret == True:
                    bounding_boxes, landmarks = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:

                        # if you want to classify about all face which is detected in frame
                        for face_no in range(nrof_faces):
                            bb_box = bounding_boxes[face_no, 0:4]
                            landmark = landmarks[:, face_no]

                        # if you want to classify about all face which is detected in frame
                        for face_no in range(nrof_faces):
                            bb_box = bounding_boxes[face_no, 0:4]
                            landmark = landmarks[:, face_no]

                            aligned = face_alignment(frame, args.image_size, landmark)
                            prewhitened = facenet.prewhiten(aligned)
                            img = np.expand_dims(prewhitened, 0)
                            aligned = face_alignment(frame, args.image_size, landmark)
                            prewhitened = facenet.prewhiten(aligned)
                            img = np.expand_dims(prewhitened, 0)

                            feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                            emb = sess.run(embeddings, feed_dict=feed_dict)

                            predictions = model.predict_proba(emb)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]

                            cv2.rectangle(frame, (int(bb_box[0]), int(bb_box[1])), (int(bb_box[2]), int(bb_box[3])),
                                          (0, 255, 0), 3)

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, '%s: %.3f' %(class_names[best_class_indices[0]],best_class_probabilities[0]),
                                        (int(bb_box[0]), int(bb_box[1])), font, 1, (255, 255, 255), 2)
                        #'%s' % (best_class_probabilities[0])
                    cv2.imshow('video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # TODO: run face detection for each frame if camera is openedq
            # detection -> alignment -> prewhitening -> embedding -> classify -> showing
            # while(cap.isOpened()):
            #    ret, frame = cap.read(~~
            #    if ret == True:
            #        <<detection>>
            #          face_detect(freame, parameter~)
            #         bounding_boxes / landmarks : no face, one face, or many faces ...
            #        bounding_boxes, landmarks =
            #        nrof_faces = bounding_boxes.shape[0]
            #        if nrof_faces > 0:
                        # if you want to classify about all face which is detected in frame
            #            for face_no in range(nrof_faces):
                        # get bounding box and landmark about each face from bounding boxes and landmarks
            #               <<alignment : face_alignment>>
            #                 bounding_boxes[face_no, 0:4]
            #                 landmarks[:,face_no]
            #               <<prewhiten : facenet.prewhiten>>
            #               <<embedding : session run>>
            #                sess.run(embeddings, feed_dict={})
            #               <<classify : model.predict>>
            #               cv2.rectangle(img, bounding_boxes(lefttop, rightbottom, :
            #                 cv2.putText(img, )
            #                clf.predict_proba(ebb)
            #            <<showing>> <- show whole at once
            #            imshow('frame', img)
            # break if key interrupted
            # if cv2.waitkey(10 && ==ord('q')
            # break
            # cv2.destroyAllWindows()
            # cap.release()
            # print("detection close")

def face_alignment(img, face_size, f_point):

    # TODO: face alignment (4week-1day ex)
    desired_left_eye = (0.35, 0.35)
    desired_right_eye = (0.65, 0.35)
    right_eye_center = (f_point[0], f_point[5])
    left_eye_center = (f_point[1], f_point[6])

    # TODO : Compute eyes center, angle and image scale
    eyesCenter = ((f_point[0] + f_point[1]) / 2, (f_point[5] + f_point[6]) / 2)
    angle = np.arctan2((f_point[6] - f_point[5]), (f_point[1] - f_point[0])) * 180 / np.pi
    scale = (0.3 * face_size) / (
        np.sqrt(np.add(np.power((f_point[0] - f_point[1]), 2), np.power((f_point[5] - f_point[6]), 2))))

    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    tX = face_size * 0.5
    tY = face_size * desired_left_eye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (face_size, face_size)
    output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    return output
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('output_dir', type=str, help='Directory with class probability results.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--cam_device', type=int,
        help='ID of the opened video capturing device (a camera index).', default=0)
    
    return parser.parse_args(argv)

#TRAIN /home/user01/Desktop/Class/W4-Day3 ./Model tmp_pickle.plk
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))