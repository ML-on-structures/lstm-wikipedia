import os
import pickle
import random
import numpy as np
from lstm import LSTM
from nn_base import DNN

# Number of iterations
N = 5#000

# Number of hidden layers in LSTM
M = 12

# Number of features present per revision
Nf = 14


def _train_nn_with_k_lstm_bits(data_list, k=None):
    """
    Get the items of dict of authors with each value containing:
     author, (matrix_of 0:n-1 revisions with features and quality,
      features of nth revision without quality,
      quality of nth revision
    )

    For a given number N of iterations:
        For each author now, run the author's data (0-n-1) revisions
        through the LSTM. From the LSTM extract k bits of output and
        send them along with nth revision's features to the Neural Net.
        Output from the Neural Net is then compared with our target value
        and loss is pushed back to Neural Net for backpropagation.
        The Neural net then provides \partial loss / \partial input for
        every input  to it. The returned values from it corresponding to
        the LSTM's k bits are then fed back through the LSTM to perform
        a backward using AdaDelta algorithm.

    :param data_list: This list contains all items in structure (author, (x_mat, fy, yt))
    :param k: Number of bits to be used
    :return: (Trained LSTM, Trained NNet), List of errors
    """


    # Initialize an LSTM
    lstm = LSTM()
    lstm.initialize(Nf, M)

    # Initialize a Neural Network
    nnet = DNN()

    # NN will take input from it's inputs + k bits.
    # Value of middle layer in NN is set with different experiments (can be changed)
    # NN will give 1 output (for quality of revision)
    nnet.initialize([k+12, k+12+(M/2), 1])

    iter_ctr = N
    # Perform the following for N iterations
    for iteration in range(N):

        # Shuffle the positions of data
        random.shuffle(data_list)

        # Create empty list for collecting errors, predicted outputs
        errors = np.array([])

        # Start the process for each author
        for cnt, (author, (x_mat, fy, yt)) in enumerate(data_list):

            # Ignore if target doesn't exist
            if not yt:
                continue

            # Send x features to the wikipedia_lstm and collect output in Y
            Y = lstm.forward(x_mat)

            # Set the input for NNet using k bits of Y
            nnet_input = np.concatenate((Y[:k], fy))

            # Sending input to NNet
            y = nnet.forward(nnet_input)

            # Quality normalized
            yt = 1.0 * (yt + 1.0) / 2.0

            # Measure squared loss
            e = np.sum((y - yt) ** 2)
            dy = 2.0 * (y - yt)

            # Add loss to error list
            errors = np.append(errors, e)

            # Now send the loss through NN backpropagation
            bp_res = nnet.backward_adadelta(dy)

            # Generate input for LSTM backward round
            back_el = np.zeros(Y.shape)
            bp_res = np.resize(bp_res, Y.shape)
            back_el[:k] = bp_res[:k]

            # Send the result on bit back through the LSTM
            lstm.backward_adadelta(back_el)

        # Print average error
        iter_ctr+=1
        if iter_ctr >= N/100:
            avg_error = np.average(errors)
            print"Avg Err at %r iteration, for all users: %r " % (iteration, avg_error)
            iter_ctr = 0

    return (lstm, nnet), errors



def _test_nn_with_k_lstm_bits(test_data, lstm, nnet, st=0, k=None):
    """
    Get the dict of authors as keys and values in form:
     (matrix_of 0:n-1 revisions with features and quality,
      features of nth revision without quality,
      quality of nth revision
    )

    Now pass it through LSTM and Neural net combination
    :param json_item:
    :return:
    """
    items = test_data.items()
    errors = np.array(0.0)

    print "\n\n==Validation==\n\n"
    for cnt, (author, (x_mat, fy, yt)) in enumerate(items):

        if not yt:
            continue

        Y = lstm.forward(x_mat)

        if st:
            if not k:
                k=0
            nnet_input = np.concatenate((Y[st:st+k], fy))
        else:
            nnet_input = np.concatenate((Y[:k], fy))
            st=0
            if not k:
                k=0
        #nnet_input = np.concatenate((Y[:k], fy))

        # Sending wikipedia_lstm outputs combined with last revisions features to Nnet
        y = nnet.forward(nnet_input)

        # Quality normalized
        yt = 1.0 * (yt + 1.0) / 2.0

        e = np.sum((y - yt) ** 2)
        errors = np.append(errors, e)

        print"Err for %r user : %r" %(cnt, e)

    print "Average validation error: ", np.average(errors)

    return np.average(errors)


def train_nn_using_1_lstm_bit(train_dict, store=False, picklefile=None):
    """

    :param train_dict:
    :param store:
    :param picklefile:
    :return:
    """
    train_items = train_dict.items()

    # Send for training using k as no. of bits to use
    net_result, errors = _train_nn_with_k_lstm_bits(train_items, k=1)

    # Store the trained model into a pickle if store is True
    if store:
        # Store the (wikipedia_lstm, nnet) type result into a pickle
        with open(picklefile, 'wb') as output:
            pickle.dump(net_result, output, pickle.HIGHEST_PROTOCOL)

    return net_result


def test_nn_using_1_lstm_bit(test_dict, lstm, nnet):
    """

    :param test_dict:
    :return:
    """

    # Send test data with trained model for testing
    validation_result = _test_nn_with_k_lstm_bits(test_dict, lstm, nnet, st=0, k=1)


    print "Using value 0 for the bit"
    #net_result = _combined_ops_nn_using_k_bits_test([item1], st=0,k=1, bit_val=0)
    print "Using value 1 for the bit"
    #net_result = _combined_ops_nn_using_k_bits_test([item1], st=0,k=1,bit_val=1)

    return locals()


if __name__ == "__main__":

    print "Starting"
    nnet_pickle = os.path.join(os.getcwd(), 'data', 'nnet_pickle.pkl')

    # Read back the pickle to act as validation during use
    with open(nnet_pickle, 'rb') as input:
        net_result = pickle.load(input)