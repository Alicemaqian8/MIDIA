import numpy as np


def train_noise_gen(train, miss_indicator):
    train_indicator = np.zeros(train.shape, int)
    chose_range = np.arange(train_indicator.shape[0])
    proportion = train.shape[0] / miss_indicator.shape[0]
    if proportion > 1:
        num = int(train.shape[0] / miss_indicator.shape[0])
        for indicator in miss_indicator:
            tmp_range = np.array(chose_range)
            tmp_perm = np.random.choice(tmp_range, num, replace=False)
            train_indicator[tmp_perm] = indicator
            chose_range = np.array(list(set(chose_range) - set(tmp_perm)))
    else:
        num = 1
        for indicator in miss_indicator:
            random = np.random.random()
            if random < proportion and chose_range.size > 0:
                tmp_range = np.array(chose_range)
                tmp_perm = np.random.choice(tmp_range, num, replace=False)
                train_indicator[tmp_perm] = indicator
                chose_range = np.array(list(set(chose_range) - set(tmp_perm)))
    train_noise = train * (1 - train_indicator)
    return train_noise, train_indicator


class DataSet(object):
    def __init__(self, input, input_noise, input_indicator):
        self._input = input
        self._input_noise = input_noise
        self._input_indicator = input_indicator
        self._num_examples = input.shape[0]  # the number of examples
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def input(self):
        return self._input

    @property
    def input_noise(self):
        return self._input_noise

    @property
    def input_indicator(self):
        return self._input_indicator

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # shuffle for the first epoch,that is disorganize the element in input
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._input = self.input[perm0]
            self._input_noise = self._input_noise[perm0]
            self._input_indicator = self._input_indicator[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished eposch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            input_rest_part = self._input[start:self._num_examples]
            input_noise_rest_part = self._input_noise[start: self._num_examples]
            input_indicator_rest_part = self._input_indicator[start: self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._input = self.input[perm]
                self._input_noise = self._input_noise[perm]
                self._input_indicator = self._input_indicator[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            input_new_part = self._input[start:end]
            input_noise_new_part = self._input_noise[start:end]
            input_indicator_new_part = self._input_indicator[start:end]
            return np.concatenate((input_rest_part, input_new_part), axis=0), np.concatenate(
                (input_noise_rest_part, input_noise_new_part), axis=0), np.concatenate(
                (input_indicator_rest_part, input_indicator_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._input[start:end], self._input_noise[start:end], self._input_indicator[start:end]