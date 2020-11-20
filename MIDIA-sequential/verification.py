# -*- coding: utf-8 -*-

'''
suppose there is only one miss dimension
'''

import numpy as np

class Verify(object):

    def __init__(self, impRes, test, miss_indicator):
        self.impRes = impRes
        self.test = test
        self.miss_indicator = miss_indicator

    def rmseCal(self):
        miss_attrs = np.where(np.sum(self.miss_indicator, axis=0) > 0)[0]
        for miss_attr in miss_attrs:
            sumErr = np.sum(np.square((self.impRes[:, miss_attr] - self.test[:, miss_attr]) * self.miss_indicator[:, miss_attr]))
            rmse = np.sqrt(sumErr / np.sum(self.miss_indicator[:, miss_attr]))
        sumErr = np.sum(np.square((self.impRes - self.test) * self.miss_indicator))
        rmse = np.sqrt(sumErr/np.sum(self.miss_indicator))
        return rmse



