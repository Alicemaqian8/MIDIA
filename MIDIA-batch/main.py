# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import denoiseAutoencoder
import mvsImputation
import verification
import input


if __name__ == "__main__":

    # AirQuality dataset
    data_dir = "../dataset/AirQuality/"
    total_attrs_num = 8

    # Parameters
    learning_rate = 0.01
    training_epochs = 1000
    batch_size = 256
    # Network Parameters
    n_input = 8  # data input
    n_hidden_1 = 5  # 1st layer num features

    # #############training####################
    # train and test dataset, the datatype is ndarray, has been normalized
    train = np.loadtxt(data_dir + "train_data.txt", delimiter=",")
    test = np.loadtxt(data_dir + "test_data.txt", delimiter=",")
    test_indicator = np.loadtxt(data_dir + "test_indicator.txt", delimiter=",")  # 1 is missing

    # train_noise generation
    train_noise, train_indicator = input.train_noise_gen(train, test_indicator)
    #np.savetxt(data_dir + "train_noise.txt", train_noise, delimiter=",", fmt='%s')
    test_noise = (1 - test_indicator) * test
    #np.savetxt(data_dir + "test_noise.txt", test_noise, delimiter=",", fmt='%s')

    sumRMSE = 0
    count = 1
    for i in range(count):
        ##############training phase####################
        dAE_model = denoiseAutoencoder.DenoiseAutoencoder(train, train_noise, train_indicator, n_input, n_hidden_1,
                                                          learning_rate, training_epochs, batch_size)
        dAE_model.tensorFlowPro()
        weights = dAE_model.getWeights()
        biases = dAE_model.getBiases()
        ##############testing phase####################
        mvsImp = mvsImputation.Imputation(test_noise, n_input, weights, biases)
        impRes, hidden = mvsImp.missingValImp()
        ##############Verification#########
        resVry = verification.Verify(impRes, test, test_indicator)
        rmse = resVry.rmseCal()
        sumRMSE += rmse
        print("rmse: ", rmse)

    avgRMSE = sumRMSE / count
    print("avgRMSE: ", avgRMSE)





