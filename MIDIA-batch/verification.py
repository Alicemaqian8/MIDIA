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
        sumErr = np.sum(np.square((self.impRes - self.test) * self.miss_indicator))
        rmse = np.sqrt(sumErr/np.sum(self.miss_indicator))
        return rmse



