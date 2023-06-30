import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test][i_train] = np.sqrt(np.sum((self.train_X[i_train] ** 2) + (X[i_test] ** 2)
                                                        - 2 * self.train_X[i_train] * X[i_test]))
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        A = self.train_X ** 2
        for i_test in range(num_test):
            dists[i_test] = np.sqrt((A + (X[i_test] ** 2)).sum(axis=1) - (2 * self.train_X.dot(X[i_test].T)))
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        s1 = (X ** 2).sum(axis=1)
        s2 = (self.train_X ** 2).sum(axis=1)
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        dists = np.sqrt(s1.reshape(-1, 1) + s2 - 2 * X.dot(self.train_X.T))
        return dists


    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        replica_dists = dists.copy()
        pred = np.zeros(num_test, bool)
        for i in range(num_test):
            arr = []
            for j in range(self.k):
                idx = np.argmin(replica_dists[i])
                #print(idx)
                arr = np.append(arr, np.array(self.train_y[idx]))
                replica_dists[i][idx] = 999999
            t = np.count_nonzero(arr)
            f = len(arr) - t
            if (t > f):
                pred[i] = True
            elif (f > t):
                pred[i] = False
            else:
                r = np.random.randint(2)
                if (r == 0):
                    pred[i] = True
                else:
                    pred[i] = False
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, int)
        replica_dists = dists.copy()
        for i in range(num_test):
            arr = []
            for j in range(self.k):
                idx = np.argmin(replica_dists[i])
                arr = np.append(arr, self.train_y[idx])
                replica_dists[i][idx] = 999999

            mas = np.zeros(10, dtype=np.int32)
            for l in range(len(arr)):
                mas[int(arr[l])] += 1
            pred[i] = np.argmax(mas)
        return pred
