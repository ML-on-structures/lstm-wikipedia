import os
import json

import pickle

from serializer import json_to_data


def load_data(files=False):
    """
    Get data from DB if not available in files.
    Otherwise use files to load data into dict objects

    :param files:
    :return:
    """
    print "Loading data"

    trn_file = os.path.join(os.getcwd(), 'data', 'training_data.json')
    tst_file = os.path.join(os.getcwd(), 'data', 'test_data.json')
    try:
        f = open(trn_file, 'r')
        vf = open(tst_file, 'r')
    except:
        print"Need to generate data from DB"
        return None

    # Read data and send to processing
    training = json_to_data(json.load(open(trn_file, 'r')))
    test = json_to_data(json.load(open(tst_file, 'r')))
    print "Data loaded: %r Training elements, and %r Test elements" % (len(training.items()), len(test.items()))
    return training, test


def get_relative_ratios(data_dict, include_all=False):
    """

    :param data_dict:
    :return:
    """
    pos = 0
    neg = 0
    items = data_dict.items()
    for cnt, (author, (x_mat, fy, yt)) in enumerate(items):
        if include_all:
            for i in x_mat:
                if i[-1] > 0:
                    pos += 1
                else:
                    neg += 1
        if yt > 0:
            pos += 1
        else:
            neg += 1

    return pos, neg


def operator_on_data(data_dict, include_all=False):
    """

    :param data_dict:
    :return:
    """
    pos = 0
    neg = 0
    items = data_dict.items()
    gooditems = [(author, (x_mat, fy, yt)) for cnt, (author, (x_mat, fy, yt)) in enumerate(items) if len(x_mat) > 10]
    print gooditems[15]
    print len(gooditems[15][1][0])


if __name__ == "__main__":
    training, test = load_data(files=True)

    #
    #
    # postive, negative = get_relative_ratios(test, include_all=True)
    # ratio = negative*1.0/(postive+negative)*1.0
    # postive, negative = get_relative_ratios(test)
    # ratio_only_last = negative*1.0/(postive+negative)*1.0
    #
    # print postive, negative, ratio
    # print "Neg/All for only last revisoins of each user", ratio_only_last, 2*(ratio_only_last)*(1-ratio_only_last)
    # print "Neg/All for all revisions", ratio, 2*(ratio)*(1-ratio)
    N = 1000
    test_only = False
    weighted_learning = True
    picklefile = os.path.join(os.getcwd(), 'data',
                              'trained_lstm_%r_%r.pkl' % ("weighted" if weighted_learning else "unweighted", N))

    from training import train_nn_using_k_lstm_bit, test_nn_using_1_lstm_bit

    if test_only:
        with open(picklefile, 'rb') as input:
            (lstm, nn) = pickle.load(input)
        test_result = test_nn_using_1_lstm_bit(test, lstm, nn)
    else:
        (lstm, nn) = train_nn_using_k_lstm_bit(training, k=1, store=True, N=N, picklefile=picklefile,
                                               weighted_learning=weighted_learning)
        test_result = test_nn_using_1_lstm_bit(test, lstm, nn)
