"""
This file is for the functions that will analyze a graph generated
for a wiki using user_graph.

The graph can be for any language wiki.

Functions in this file should build labels from a graph,
build required LSTM models and run them.
"""
import json
import os
import random
from pprint import pprint
from datetime import datetime
from json_plus import Serializable
from multi_layer_lstm.multi_layer_LSTM import Instance_node, Multi_Layer_LSTM

import numpy as np

WIKINAME = 'astwiki'
# WIKINAME = 'rmywiki'

DEFAULT_PARAMS = {}

FEATURE_VECTOR_SIZE = 8


def _timediff(timestamp1, timestamp2):
    """
    Returns number of seconds between two timestamp values
    :param timestamp1:
    :type timestamp1:
    :param timestamp2:
    :type timestamp2:
    :return:
    :rtype:
    """
    # return (datetime.fromtimestamp(timestamp1) - datetime.fromtimestamp(timestamp2)).total_seconds()
    return (timestamp1 - timestamp2).total_seconds()


def build_labels_for_revision_quality(wikidata):
    """

    :param wikidata:
    :type wikidata:
    :return:
    :rtype:
    """
    l2_list = [len(v) for k, v in wikidata.iteritems()]

    min_l2 = min(l2_list)
    max_l2 = max(l2_list)
    avg_l2 = np.average(l2_list)

    print min_l2, max_l2, avg_l2
    print l2_list


def _days(v):
    return v / 3600 / 24

def _secs_in_days(d):
    return d*3600*24


def build_labels_for_user_quitting(user_contribs):
    """

    From the set of user contributions, get the latest timestamp available in database.
    Then from this latest timestamp, measure a distance upto the
    latest by each user, and establish a threshold.

    :param user_contribs:
    :type user_contribs:
    :return:
    :rtype:
    """
    timestamps = {}

    for user, record in user_contribs.iteritems():
        latest_time = max([int(i['timestamp']) for i in record])
        timestamps[user] = datetime.fromtimestamp(latest_time)

    pprint(timestamps)
    print len(timestamps)
    maxt = max(timestamps.values())

    delta_dict = {k: _timediff(maxt, v) for k, v in timestamps.iteritems()}

    pprint(delta_dict)

    l2_list = delta_dict.values()

    min_l2 = min(l2_list)
    max_l2 = max(l2_list)
    avg_l2 = np.average(l2_list)

    print min_l2, max_l2, avg_l2
    histo = np.histogram(l2_list)

    print histo
    for i in histo[1]:
        print _days(i)

    quit_labels = {k: 0 if _days(v) > 40 else 1 for k, v in delta_dict.iteritems()}

    print np.histogram(quit_labels.values())


def _create_label_file(labels, name_suffix):
    filename = os.path.join(os.getcwd(), 'results', WIKINAME, 'labels_%s.json' % (name_suffix))
    with open(filename, 'wb') as outp:
        outp.write(Serializable.dumps(labels))
    pass


def _print_distribution(labels):
    len_0s = len([i for i in labels.values() if i == 0])
    len_1s = len([i for i in labels.values() if i == 1])
    len_list = 1.0 * len(labels)

    def _percent(len):
        return 100 * (1.0 * len / len_list)

    print "Distribution 1s: %d and 0s: %d" % (len_0s, len_1s)
    print "Percentage 1s: %0.2f and 0s: %0.2f" % (_percent(len_0s), _percent(len_1s))


def user_quitting_labels(user_graph):
    """

    :param user_graph:
    :type user_graph:
    :return:
    :rtype:
    """

    MAX_DAYS = 1
    print len(user_graph)
    # user_graph = _graph_from_days_back(graph=user_graph, ending=100, starting_from = 5000)

    print len(user_graph)


    max_time = max([max([i['timestamp'] for k, i in a.iteritems()]) for t, a in user_graph.iteritems()])

    quit_labels = {k: 0 if _days(max_time - max([i['timestamp'] for _, i in v.iteritems()])) < MAX_DAYS else 1 for k, v
                   in
                   user_graph.iteritems()}

    _create_label_file(labels=quit_labels, name_suffix="quit")

    _print_distribution(labels=quit_labels)

    return quit_labels


def _qual(d01, d12, d02):
    return (1.0 * (d02 - d12) / 1.0 * d01)


def _reverts(v):
    return True if _qual(d01=v['d01'], d12=v['d12'], d02=v['d02']) < 1 else False


def _graph_from_days_back(graph, ending=7, starting_from=None):
    """
    Return the state of graph as it would have been days before.
    Basically in the entire graph, remove entries of each author
    contributed after a certain number of days from top

    :param graph:
    :type graph:
    :param ending:
    :type ending:
    :return:
    :rtype:
    """
    max_time = max([max([i['timestamp'] for k, i in a.iteritems()]) for t, a in graph.iteritems()])
    check_time = max_time - _secs_in_days(ending)

    if starting_from is None:
        return {user:{k:v for k,v in values.iteritems() if v['timestamp']<check_time} for user,values in graph.iteritems()}

    else:
        pop_list = []
        previous_time = max_time - _secs_in_days(starting_from)
        ret_graph = {user:{k:v for k,v in values.iteritems() if v['timestamp']<check_time and v['timestamp']>previous_time}  for user,values in graph.iteritems()}

        pop_list = [user for user, values in ret_graph.iteritems() if len(values)==0]

        for i in pop_list:
            ret_graph.pop(i)

        return ret_graph


def user_reversion_label(user_graph):
    """

    Reversion means that the user's revision was reverted by next editor.
    So the label based on latest revision by an author goes like this:
        If the q on user's last revision <0, then reverted

    :param user_graph:
    :type user_graph:
    :return:
    :rtype:
    """

    pop_list = []

    user_graph = _graph_from_days_back(graph=user_graph, ending=14)

    user_last_revs = {k: max(v.values(), key=lambda x: x['timestamp']) if len(v) else pop_list.append(k) for k, v in user_graph.iteritems()}

    # print pop_list
    for i in pop_list:
        user_last_revs.pop(i)

    reversion_label = {k: 1 if _reverts(v['list'][0]) else 0 for k, v in user_last_revs.iteritems()}

    _create_label_file(labels=reversion_label, name_suffix="reversion")

    _print_distribution(labels=reversion_label)

    return reversion_label


def learn_from_graph(user_graph, user_labels, params=DEFAULT_PARAMS):
    """

    :param user_graph:
    :type user_graph:
    :param user_labels:
    :type user_labels:
    :param params:
    :type params:
    :return:
    :rtype:
    """


def _dist_normal(distance):
    return distance/100.0


def _normalize_vector_for_features(element, feature_vector_size):

    TIME_DENOMINATOR = 100000.0

    fv = np.zeros(feature_vector_size)

    # Features

    # Distance 01
    fv[0] = _dist_normal(element['d01'])
    # Distance 02
    fv[1] = _dist_normal(element['d02'])
    # Distance 12
    fv[2] = _dist_normal(element['d02'])

    # Time 01
    fv[3] = np.log(1+element['t01']/TIME_DENOMINATOR)
    # Time 12
    fv[4] = np.log(1+element['t12']/TIME_DENOMINATOR)

    # Delta
    fv[5] = _dist_normal(element['Delta'])
    # dp2
    fv[6] = _dist_normal(element['dp2'])

    # Quality calculated by 1 on 0
    fv[7] = _qual(d01=element['d01'],d02=element['d02'],d12=element['d12'])

    return fv


def _get_features_of_edit(data, feature_vector_size = FEATURE_VECTOR_SIZE):
    """

    :param data:
    :type data:
    :return:
    :rtype:
    """
    if data.has_key('list'):
        if len(data['list']):
            element = data['list'][0]
            return _normalize_vector_for_features(element, feature_vector_size)
        else:
            return np.zeros(feature_vector_size)
    else:
        return None


def build_entries_for_learning(user_graph, labels, max_depth):
    """
    Build the structure using Instance nodes form multi_layer_LSTM

    From the graph of users, we need to build training trees of each
    user and put some in training and some in test

    Grpah looks like:
    {'user0':{'rev0':[{'user00'}],
                'rev1':[{user01}]
                },
     'user1':{'rev0':[{'user10'}],
                'rev1':[{user11}]
                },
    }

    :param user_graph:
    :type user_graph:
    :param max_depth:
    :type max_depth:
    :return:
    :rtype:
    """
    instance_list = []

    for user, values in user_graph.iteritems():
        if not labels.has_key(user):
            continue
        new_node = Instance_node(label=labels[user])
        for edit,data in values.iteritems():
            new_child = Instance_node()
            new_child.feature_vector = _get_features_of_edit(data)
            if new_child.feature_vector is not None:
                new_node.children.append(new_child)

        instance_list.append(new_node)

    return instance_list



class GraphLearning(Serializable):
    """
    Class for the learning models

    Learning is as follows:

        Iterate over a shuffled set of users
            For each user, over all the user's contributions,
            we get the first edit on it by another person.
            This edit defines
    """

    def __init__(self, user_graph, user_labels):
        """

        :param user_graph:
        :type user_graph:
        :param user_labels:
        :type user_labels:
        """

        self.user_graph = user_graph
        self.user_labels = user_labels

    def initialize_settings(self, params):
        """
        Initialize the settings like LSTMs etc
        depending on the values in parameter set.

        :param params:
        :type params: dict
        :return:
        :rtype:
        """

    def _get_user_revisions(self, user, timestamp):
        pass

    def _compute_features_for_lstm(self, item, depth_now):
        """
        This function gets called recursively by itself to get the
        LSTM features for an entry through each layer going down.

        For each revision, its revision features can be computed right away,
        but the LSTM features are an output from LSTM on the lower layer.
        So to fill those, we need to send a recursive call to this function with the
        item being user for whom we run the lower layer LSTM.

        When we reach the max depth parameter, we return only the set of revision
        features passed through LSTM on that layer and not the concatenation of
        revision features and previous layer LSTM features

        :param item:
        :type item:
        :return:
        :rtype:
        """
        node_data_list = self._get_user_revisions(item, self._get_item_timestamp(item))

        if depth_now < self.params['max_depth']:

            # Initialize LSTM input matrix entry for user

            for instance in node_data_list:
                previous_layer_lstm_features = self._compute_features_for_lstm(instance['user2'], instance['timestamp'], depth_now + 1)
                entry_features = _get_rev_features(instance)

                entry_mat = _generate_matrix_for_depth()

                return self.LSTMs[depth_now].forward()


        else:
            return None

            # Matrix row for LSTM input here combines rev features and graph features together

    def initiate_training_epoch(self):
        """
        This function runs the iterator which calls the recursive function
        to generate LSTM output features for each entry

        :return:
        :rtype:
        """
        graph_item_list = self.user_graph.items()
        random.shuffle(graph_item_list)

        for item in graph_item_list:
            self._compute_features_for_lstm(item)



if __name__ == "__main__":
    # Get the data for concerned wiki

    graph_file = os.path.join(os.getcwd(), 'results', WIKINAME, 'reduced_user_graph.json')

    with open(graph_file, 'rb') as inp:
        wikidata = Serializable.load(inp)

    # contrib_file = os.path.join(os.getcwd(), 'results', 'user_contrib_test.json')
    #
    # with open(contrib_file, 'rb') as inp:
    #     user_contribs = json.load(inp)

    # Build labels

    # build_labels_for_user_quitting(user_contribs=user_contribs)
    # build_labels_for_revision_quality(wikidata=wikidata)

    labels = user_quitting_labels(wikidata)
    # labels = user_reversion_label(wikidata)
