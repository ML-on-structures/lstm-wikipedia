import os
import json
import numpy as np
import pickle
from db import DataAccess
from serializer import json_to_data


def load_data(files=False, trn_file='training_data.json', tst_file='test_data.json'):
    """

    Get results from DB if not available in files.
    Otherwise use files to load results into dict objects

    :param files:
    :param trn_file:
    :param tst_file:
    :return:
    """
    print "Loading results"

    trn_file = os.path.join(os.getcwd(), 'results', trn_file)
    tst_file = os.path.join(os.getcwd(), 'results', tst_file)
    try:
        f = open(trn_file, 'r')
        vf = open(tst_file, 'r')
    except:
        print"Need to generate results from DB"
        db = DataAccess()
        training_from_db, test_from_db = db.load_fresh_from_db()
        return training_from_db, test_from_db

    # Read results and send to processing
    training_from_json = json_to_data(json.load(open(trn_file, 'r')))
    test_from_json = json_to_data(json.load(open(tst_file, 'r')))
    print "Data loaded: %r Training elements, and %r Test elements" % (
        len(training_from_json.items()), len(test_from_json.items()))
    return training_from_json, test_from_json


def create_user_last_revision_existence_data(data_dict):
    """
    From training and test results sets, prepare a new dataset
    with labels for predicitng existence of next revisions of an author.

    The logic used here goes like this:
     If revisions are over 30 and std between consecutive revision
     times is low, then check the time gap from
    :return:
    """
    new_items = {}
    data_to_use = data_dict.copy()
    items = data_to_use.items()
    print "Items: ", len(items)
    print "Shape before-- x_mat: %r, fy: %r" % (items[0][1][0].shape, items[0][1][1].shape)

    largest_xmat = max([len(x_mat) for (author, (x_mat, fy, yt)) in items])
    print "Largest xmat ", largest_xmat
    for cnt, (author, (x_mat, fy, yt)) in enumerate(items):
        # Look at the time from previous user contribution.
        # If that time is greater than threshold,
        # then label for this revision coming in is 0. Otherwise 1

        x_mat = np.array([np.concatenate((i,[0])) if i[0]>0.3 else np.concatenate((i,[1])) for i in x_mat])
        #fy = np.concatenate((fy, np.array([0,yt])))
        yt = 0.0 if fy[0]>0.0005 else 1.0

        # # Build solely on truth
        # # So every entry from 0- (n-1) gets 0 label meaning it is not the last revisions
        # x_mat = np.array([np.concatenate((i, [0])) for i in x_mat])
        #
        # # Last entry gets 0 only if x_mat was of size 49 in size, otherwise 1
        # yt = 0 if len(x_mat) == 49 else 1

        # Append to new list
        new_items[author] = (x_mat, fy, yt)

    print "New Items: ", len(new_items)

    return new_items


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

    objective = "quality"
    # objective = "existence"

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
    from learning import train_nn_using_k_lstm_bit, test_nn_using_k_lstm_bit, train_nn_only, test_nn_only, test_random

    N = 5000
    k = 12
    test_only = False
    weighted_learning = False
    balanced = True

    if objective == "quality":
        picklefile = os.path.join(os.getcwd(), 'results',
                                  'trained_lstm_k%r_%r_%r_%r.pkl' % (
                                      k, "weighted" if weighted_learning else "unweighted",
                                      "balance" if balanced else "unbalanced", N))
        nn_pickle = os.path.join(os.getcwd(), 'results',
                                 'trained_nn_only_%r.pkl' % (N))

        if test_only:
            with open(picklefile, 'rb') as input:
                (lstm, nn) = pickle.load(input)
            test_result = test_nn_using_k_lstm_bit(test, lstm, nn, k=1)
            with open(nn_pickle, 'rb') as input:
                (lstm, nn) = pickle.load(input)
            test_result = test_nn_only(test, lstm, nn)
        else:
            (lstm, nn) = train_nn_using_k_lstm_bit(training, k=k, store=True, N=N, picklefile=picklefile,
                                                   weighted_learning=weighted_learning, balanced=balanced)
            test_result = test_nn_using_k_lstm_bit(test, lstm, nn, k=k)

            # (lstm, nn) = train_nn_only(training, store=True, N=N, picklefile=nn_pickle,
            #                             weighted_learning=weighted_learning)
            # test_result = test_nn_only(test, lstm, nn)

            # print "Oracle Round"
            # oracle_result = test_oracle(test)

    elif objective == "existence":
        picklefile = os.path.join(os.getcwd(), 'results',
                                  'existence_trained_lstm_%r_%r.pkl' % (
                                      "weighted" if weighted_learning else "unweighted", N))
        nn_pickle = os.path.join(os.getcwd(), 'results',
                                 'existence_trained_nn_only_%r.pkl' % (N))
        training_new = create_user_last_revision_existence_data(training)
        test_new = create_user_last_revision_existence_data(test)

        (lstm, nn) = train_nn_using_k_lstm_bit(training_new, k=k, store=True, N=N, picklefile=picklefile,
                                               weighted_learning=weighted_learning, quality=False)
        test_result = test_nn_using_k_lstm_bit(test_new, lstm, nn, k=k, quality=False)
