'''
Classes and functions for recognition of digits and english alphabet
Written by stuharvey with credits to github/abidrahmank
http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_ml/
       py_knn/py_knn_opencv/py_knn_opencv.html'''
import numpy as np
import cv2
import os
import sys


class Digits:
    def __init__(self):
        self.img = cv2.imread('digits.png')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # convert each example to an element of a list of 20x20px matrices
        ''' following code merits some explanation
            - np.vsplit(img, 50) splits img into 50 rows, hsplit similar
            - the code below is equivalent to
                    digits = []
                    rows = np.vsplit(img, 50)
                    for row in rows:
                        digits.append(np.hsplit(row,100))'''
        self.digits = [np.hsplit(row, 100) for row in np.vsplit(self.img, 50)]
        # convert list to matrix(rows=50, cols=50, dtype=(20x20 ndarray))
        self.digits = np.array(self.digits)

        self.train = None
        self.train_labels = None

        if not os.path.isfile('digits_knn_train.npz'):
            self.train_knn()

    # find digit of arbitrary input (should be appropriate nparray)
    # scale to 20x20
    def get_label(self, input):
        # test = __sanitize_for_knn(input)
        self.train, self.train_labels = self.__load_training()
        self.knn = cv2.KNearest()
        self.knn.train(self.train, self.train_labels)

        self.ret, self.result, self.neighbors, self.dist = \
            self.knn.find_nearest(input, k=5)
        return self.result

    # for now manually sanitize before call to get_label,
    # need to decide how live ocr will work before writing this
    def __sanitize_for_knn(input):
        return input

    # would like to do this for arbitrary digits but not right now
    def train_knn(self):
        # build training and test set 50/50
        ''' training set is every column 0-49, reshaped into a list '''
        self.trainer = self.digits[:, :80].reshape(-1, 400).astype(np.float32)
        self.tester = self.digits[:, 80:100].reshape(-1, 400).\
            astype(np.float32)

        # create label
        self.digs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.train_labels = np.repeat(self.digs, 400)[:, np.newaxis]
        # use test labels to calculate accuracy of model
        self.test_labels = np.repeat(self.digs, 100)[:, np.newaxis]

        # we use K-nearest-neighbors to classify digits
        self.knn = cv2.KNearest()
        self.knn.train(self.trainer, self.train_labels)
        ''' knn.train creates a cartesian map of all sample digits
            ---
            the following evaluation calculates the euclidean distance
            from each test digit to each sample digit
            given k=5, it examines the 5 lowest distances from a test
            digit, counting the labels of each distance. whichever label
            occurs most often is the "nearest neighbor", so the test digit
            is labelled in that category. '''
        self.ret, self.result, self.neighbors, self.dist = \
            self.knn.find_nearest(self.tester, k=5)

        # calculate accuracy (93.8 %):
        self.is_match = self.result == self.test_labels
        self.correct = np.count_nonzero(self.is_match)
        self.accuracy = self.correct*100.0/self.result.size

        # save to file
        try:
            np.savez('digits_knn_train.npz', train=self.trainer, 
                     train_label=self.train_labels)
        except:
            print("Error writing training set: ", sys.exc_info()[0])

    # load training set
    def __load_training(self):
        with np.load('digits_knn_train.npz') as self.training:
            self.train = self.training['train']
            self.train_labels = self.training['train_labels']
            return self.train, self.train_labels


class Characters:
    def __init__(self):
        self.alph = np.loadtxt('letter-recognition.data', dtype='float32',
                               delimiter=',', converters=
                               {0: lambda ch: ord(ch)-ord('A')})
        if not os.path.isfile('chars_knn_train.npz'):
            self.train_knn()

    # returns the class the input is given
    def get_label(self, input):
        self.train, self.train_labels = self.__load_training()
        self.knn = cv2.KNearest()
        self.knn.train(self.train, self.train_labels)

        self.ret, self.result, self.neighbors, self.dist = \
            self.knn.find_nearest(input, k=5)
        return self.result

    def train_knn(self):
        self.train, self.test = np.vsplit(self.alph, 2)
        self.train_labels, self.train_features = np.hsplit(self.train, [1])
        self.test_labels, self.test_features = np.hsplit(self.train, [1])

        # do classification
        self.knn = cv2.KNearest()
        self.knn.train(self.train_features, self.train_labels)
        self.ret, self.result, self.neighbours, self.dist = \
            self.knn.find_nearest(self.test_features, k=5)

        # calc accuracy
        self.correct = np.count_nonzero(self.result == self.test_labels)
        self.accuracy = self.correct*100.0/self.result.size

        # save training set
        try:
            np.savez('chars_knn_train.npz', train=self.train_features,
                     train_label=self.train_labels)
        except:
            print("Error writing training set: ", sys.exc_info()[0])

    def __load_training(self):
        with np.load('chars_knn_train.npz') as self.training:
            self.train = self.training['train_features']
            self.train_labels = self.training['train_labels']
            return self.train, self.train_labels
