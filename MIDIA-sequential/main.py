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
    mean_arr = np.array([0.325, 0.3037, 0.2174, 0.407, 0.3483, 0.4347, 0.5036, 0.4109])

    # #############training####################
    # train and test dataset, the datatype is ndarray, has been normalized
    train = np.loadtxt(data_dir + "train_data.txt", delimiter=",")
    test = np.loadtxt(data_dir + "test_data.txt", delimiter=",")
    test_indicator = np.loadtxt(data_dir + "test_indicator.txt", delimiter=",")  # 1 is missing

    # train_noise generation
    train_indicator = input.indicator_gen(train, test_indicator)
    train_noise = input.noise_gen(train, train_indicator, mean_arr)
    # np.savetxt(data_dir + "train_noise.txt", train_noise, delimiter=",", fmt='%s')
    # test_noise = (1 - test_indicator) * test
    test_noise = input.noise_gen(test, test_indicator, mean_arr)
    # np.savetxt(data_dir + "test_noise.txt", test_noise, delimiter=",", fmt='%s')

    # miss attrs find
    all_attrs = np.arange(total_attrs_num)
    miss_nums = np.sum(test_indicator, axis=0)  # sum by column
    miss_attrs = np.where(miss_nums > 0)[0]
    miss_indexes_sort = np.argsort(miss_nums[miss_attrs])
    cmp_attrs = np.array(list(set(all_attrs) - set(miss_attrs)))
    print("miss_nums: ", miss_nums)
    print("miss_attrs: ", miss_attrs, "cmp_attrs: ", cmp_attrs)

    impRes = np.array(test_noise)
    for miss_index in miss_indexes_sort:
        miss_attr = miss_attrs[miss_index]
        tmp_all_attrs = np.concatenate((np.array([miss_attr]), cmp_attrs))
        tmp_train = train[:, tmp_all_attrs]
        tmp_train_noise = train_noise[:, tmp_all_attrs]
        tmp_train_indicator = train_indicator[:, tmp_all_attrs]
        tmp_n_input = tmp_all_attrs.size
        tmp_n_hidden_1 = tmp_n_input - 2
        ##############training phase####################
        dAE_model = denoiseAutoencoder.DenoiseAutoencoder(tmp_train, tmp_train_noise, tmp_train_indicator, tmp_n_input,
                                                          tmp_n_hidden_1,
                                                          learning_rate, training_epochs, batch_size)
        train_impRes = dAE_model.tensorFlowPro()
        weights = dAE_model.getWeights()
        biases = dAE_model.getBiases()
        train_miss_perm = np.where(tmp_train_indicator[:, 0] == 1)[0]
        train_noise[train_miss_perm, miss_attr] = train_impRes[train_miss_perm, 0]
        cmp_attrs = np.concatenate((cmp_attrs, np.array([miss_attr])))
        ##############testing phase####################
        tmp_test_noise = test_noise[:, tmp_all_attrs]
        tmp_mvsImp = mvsImputation.Imputation(tmp_test_noise, tmp_n_input, weights, biases)
        tmp_impRes, tmp_hidden = tmp_mvsImp.missingValImp()
        test_miss_perm = np.where(test_indicator[:, miss_attr] == 1)[0]
        impRes[test_miss_perm, miss_attr] = tmp_impRes[test_miss_perm, 0]
        test_noise[test_miss_perm, miss_attr] = tmp_impRes[test_miss_perm, 0]
    ##############Verification#########
    resVry = verification.Verify(impRes, test, test_indicator)
    rmse = resVry.rmseCal()
    print("rmse: ", rmse)










