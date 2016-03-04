import os
import random

import pickle
from scipy import stats

import numpy as np

from basics import load_data
from training import test_nn_using_k_lstm_bit
import matplotlib.pyplot as plt

FEATURES = {
    '0': "time from previous edit by user",
    '1': "time from previous edit on this page",
    '2': "time from previous edit by user on this page",
    '3': "characters added",
    '4': "characters removed",
    '5': "spread",
    '6': "position in page",
    '7': "time in day",
    '8': "day of week",
    '9': "length of comment after review",
    '10': "ration  of uppercase/lowercase",
    '11': "digit/total ratio",
    '12': "time to next revision",
    '13': "quality of revision",
}


def _return_bit_values(test_data, lstm, nnet, k=None, quality=True):
    """

    Get the dict of authors as keys and values in form:
     (matrix_of 0:n-1 revisions with features and quality,
      features of nth revision without quality,
      quality of nth revision
    )

    Now pass it through trained LSTM and Neural net combination which has been
    For each author,
        run the author's data (0-n-1) revisions
        through the LSTM. From the LSTM extract k bits of output and
        send them along with nth revision's features to the Neural Net.
        Output from the Neural Net is then compared with our target value
        and loss is reported.
        Add the loss, true label and predicted label to respective lists
        Also add a weighing factor to list of weights which corresponds
        to the size of update

    :param test_data: Dict containing items in structure (author, (x_mat, fy, yt))
    :param lstm: Trained LSTM
    :param nnet: Trained Neural Net
    :param k: Number of bits to be used
    :return: Errors, True labels, Predicted labels
    """

    # Get items from dict
    items = test_data.items()

    # Create empty lists for errors, lables and weights
    errors = np.array([])
    y_true = np.array([])
    y_pred = np.array([])
    label_weights = np.array([])

    bits_to_use = []

    print "\n\n==Validation==\n\n"

    # Start the process for each author
    for cnt, (author, (x_mat, fy, yt)) in enumerate(items):

        # Check to ignore entry in absence of target value
        if not yt:
            continue
        Y = np.array([])
        if k > 0 or k is None:
            # Run LSTM only if bits from LSTM are required
            # Compute output from LSTM using 1-(n-1) revisions
            Y = lstm.forward(x_mat)

        bits_to_use = Y[:k]
        # Set the input for NNet using k bits of Y
        nnet_input = np.concatenate((Y[:k], fy))

        # Sending LSTM output bit combined with last revisions features to Nnet
        y = nnet.forward(nnet_input)

        if k > 0 or k is None and quality:
            # Quality normalized
            yt = 1.0 * (yt + 1.0) / 2.0

        # Measure error
        e = np.sum((y - yt) ** 2)

        # Append error and target entries into corresponding lists
        errors = np.append(errors, e)
        y_pred = np.append(y_pred, y)
        y_true = np.append(y_true, yt)

        # Get update size using average of this revision's
        # char added and char subtracted (fy[3] and fy[4])
        update_size = np.average((fy[3], fy[4]))
        label_weights = np.append(label_weights, update_size)

    print "Average validation error: ", np.average(errors)

    # Return the computed labels along with errors, true labels and weights
    return errors, y_pred, y_true, label_weights, bits_to_use


def meaning_of_bits(data, lstm, nn, k):
    """

    :return:
    """
    items = data.items()
    # Get elements with at least 10 entries
    refined = [(author, (x_mat, fy, yt)) for cnt, (author, (x_mat, fy, yt)) in enumerate(items) if len(x_mat) > 10]

    selected_item = random.choice(refined)

    # Operate on item1 and break it into multiple values so as to
    # get revision count increating step by step

    selected_xmat = selected_item[1][0]

    new_list = []
    for c, val in enumerate(selected_xmat):
        if c + 1 < len(selected_xmat):
            new_list.append(
                {selected_item[0]: (selected_xmat[:c + 1], selected_xmat[c + 1][:-2], selected_xmat[c + 1][-1])})

    bits_used = {}
    features = [[] for i in range(14)]

    for entry in new_list:
        errors, y_pred, y_true, label_weights, bits_to_use = _return_bit_values(entry, lstm, nn, k=k)
        #     out_0, err_0 = _check_bitvalues_of_nn([entry], netres[0],netres[1],st=0,k=1, bit_value=0)
        #     out_1, err_1 = _check_bitvalues_of_nn([entry], netres[0],netres[1],st=0,k=1, bit_value=1)
        #
        #     #print "Regular--- Output: ", out_reg, ", Error: ", err_reg
        #     # print "0 bit  --- Output: ", out_0, ", Error: ", err_0
        #     # print "1 bit  --- Output: ", out_1, ", Error: ", err_1
        #     print "-------del: ", (out_reg-out_0)
        entry = entry.items()[-1]
        print entry
        for i in range(k):
            if bits_used.has_key(i):
                bits_used[i].append(bits_to_use[i])
            else:
                bits_used[i] = [(bits_to_use[i])]

        entry_xmat_last_row = entry[1][0][-1]
        for i, v in enumerate(features):
            v.append(entry_xmat_last_row[i])

    print "Length of bits", len(bits_used[0])
    print "Length of features", len(features)

    x = range(len(bits_used[0]))
    for i, v in enumerate(features):

        for k, bits in bits_used.iteritems():
            # print v
            # print bits
            plt.plot(x, bits, label="Bit %r" % (k))
            plt.plot(x, v, label=FEATURES[str(i)])
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            plt.savefig(os.path.join(os.getcwd(), 'data', 'figs', 'bit_' + str(k) + '_feat_' + str(i) + '.png'))
            plt.close()
            s = stats.ttest_ind(bits, v)
            print "\nT-test values for bit %r and feature %r: %r" % (k, i, s)

    # print "Using value 0 for the bit"
    # net_result = _combined_ops_nn_using_k_bits_test([item1], st=0,k=1, bit_val=0)
    # print "Using value 1 for the bit"
    # net_result = _combined_ops_nn_using_k_bits_test([item1], st=0,k=1,bit_val=1)

    return locals()


if __name__ == "__main__":
    training, test = load_data(files=True)

    N = 5000
    k = 12
    test_only = False
    weighted_learning = True

    picklefile = os.path.join(os.getcwd(), 'data',
                              'trained_lstm_k%r_%r_%r.pkl' % (k, "weighted" if weighted_learning else "unweighted", N))

    known_pickle = os.path.join(os.getcwd(), 'data', 'nnet_pickle_5000.pkl')

    with open(picklefile, 'rb') as input:
        (lstm, nn) = pickle.load(input)

    # with open(known_pickle, 'rb') as input:
    #     (lstm, nn) = pickle.load(input)
    meaning = meaning_of_bits(test, lstm, nn, k=k)
