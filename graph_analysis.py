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
import time
from json_plus import Serializable
# from multi_layer_lstm.multi_layer_LSTM import Instance_node, Multi_Layer_LSTM
from multi_LSTM import InstanceNode, SequenceItem, MultiLSTM
import numpy as np

from multi_layer_lstm.multi_layer_LSTM import Instance_node

# WIKINAME = 'rmywiki'  # Very small
WIKINAME = 'astwiki'  # Medium
# WIKINAME = 'bgwiki'   # Large
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
    return d * 3600 * 24


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
    if d01:
        return (1.0 * (d02 - d12) / (1.0 * d01))
    else:
        return (1.0 * (d02 - d12))


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
        return {user: {k: v for k, v in values.iteritems() if v['timestamp'] < check_time} for user, values in
                graph.iteritems()}

    else:
        pop_list = []
        previous_time = max_time - _secs_in_days(starting_from)
        ret_graph = {
        user: {k: v for k, v in values.iteritems() if v['timestamp'] < check_time and v['timestamp'] > previous_time}
        for user, values in graph.iteritems()}

        pop_list = [user for user, values in ret_graph.iteritems() if len(values) == 0]

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

    user_last_revs = {k: max(v.values(), key=lambda x: x['timestamp']) if len(v) else pop_list.append(k) for k, v in
                      user_graph.iteritems()}

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
    return distance / 100.0


def _normalize_vector_for_features(element, feature_vector_size):
    TIME_DENOMINATOR = 10000000.0

    fv = np.zeros(feature_vector_size)

    # Features

    # Distance 01
    fv[0] = _dist_normal(element['d01'])
    # Distance 02
    fv[1] = _dist_normal(element['d02'])
    # Distance 12
    fv[2] = _dist_normal(element['d02'])

    # Time 01
    # if element['t01'] < 0:
    #     print element
    fv[3] = np.log(1 + abs(element['t01']) / TIME_DENOMINATOR)
    # fv[3] = element['t01'] / (TIME_DENOMINATOR*10)

    # Time 12
    # if element['t12'] < 0:
    #     print element
    fv[4] = np.log(1 + abs(element['t12']) / TIME_DENOMINATOR)
    # fv[4] = element['t12']/(TIME_DENOMINATOR*10)

    # Delta
    fv[5] = _dist_normal(element['Delta'])
    # dp2
    fv[6] = _dist_normal(element['dp2'])

    # Quality calculated by 1 on 0
    fv[7] = _qual(d01=element['d01'], d02=element['d02'], d12=element['d12'])

    return fv


def _get_features_of_edit(data, feature_vector_size=FEATURE_VECTOR_SIZE):
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


def build_entries_for_learning(user_graph, labels, max_depth=None):
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
        for edit, data in values.iteritems():
            new_child = Instance_node()
            new_child.feature_vector = _get_features_of_edit(data)

            # new_child.feature_vector = _get_features_of_edit(data)

            # 2nd layer stuff begins
            # print data
            new_child_node = data['list'][0]['uname2']
            if not user_graph.has_key(new_child_node):
                continue
            limit_items = {k: user_graph[new_child_node][k] for k in user_graph[new_child_node].keys()[:5]}
            for e2, d2 in limit_items.iteritems():
                new_grandchild = Instance_node()
                new_grandchild.feature_vector = _get_features_of_edit(d2)
                if new_grandchild.feature_vector is not None:
                    new_child.children.append(new_grandchild)
            # 2nd layer stuff ends
            if new_child.feature_vector is not None:
                new_node.children.append(new_child)

        instance_list.append(new_node)

    return instance_list


def _get_sequence_list_for(values, sequence_control=None, limiter=None):
    """
    Based on the sequence control, generate sequence list of SequenceItem type objects
    :param values:
    :type values:
    :param sequence_control:
    :type sequence_control:
    :return:
    :rtype:
    """
    sequence_list = []
    for edit, data in values.iteritems():
        if not len(data):
            continue
        if not data.has_key('list'):
            continue
        if not len(data['list']):
            continue
        if not labels.has_key(data['list'][0]['uname2']):
            continue

        # After all checks to not enter an item, create the SequenceItem object
        # and get its features
        link_node = data['list'][0]['uname2']
        s = SequenceItem(item_id=edit,
                         link_node_id=link_node,
                         feature_vector=_get_features_of_edit(data),
                         timestamp=data['timestamp'],
                         action_time=data['list'][0]['timestamp'])

        sequence_list.append(s)

    if sequence_control == 'time':
        return sorted(sequence_list, key=lambda x: x.timestamp)[:limiter]


def generate_instance_graph(graph_data, labels, limiter=20):
    """
    Generate the graph structure using Instance and SequenceItem classes

    instance_graph = {
        'user1':InstanceNode(label, [SequenceItem], gradient, cache, sequence_control)
    }

    :param graph_data:
    :type graph_data:dict
    :return:
    :rtype:
    """

    SEQ_CONT = 'time'
    LIMITER = limiter

    instance_graph = {}

    for user, values in graph_data.iteritems():
        if not labels.has_key(user):
            continue
        new_node = InstanceNode(label=labels[user], sequence_control=SEQ_CONT)

        # Now that label is set for node, get its sequence list
        new_node.sequence_list = _get_sequence_list_for(values, sequence_control=SEQ_CONT, limiter=LIMITER)

        instance_graph[user] = new_node

    return instance_graph


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
                previous_layer_lstm_features = self._compute_features_for_lstm(instance['user2'], instance['timestamp'],
                                                                               depth_now + 1)
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


def _f1(prec, rec):
    return (2.0*prec*rec)/(1.0*(prec+rec))


if __name__ == "__main__":
    # Get the data for concerned wiki

    graph_file = os.path.join(os.getcwd(), 'results', WIKINAME, 'reduced_user_graph.json')

    with open(graph_file, 'rb') as inp:
        wikidata = Serializable.load(inp)
        # wikidata = {k:wikidata[k] for k in random.sample(wikidata.keys(), 10000)}
        print len(wikidata)

    BREADTH = 1

    # contrib_file = os.path.join(os.getcwd(), 'results', 'user_contrib_test.json')
    #
    # with open(contrib_file, 'rb') as inp:
    #     user_contribs = json.load(inp)

    # Build labels

    # build_labels_for_user_quitting(user_contribs=user_contribs)
    # build_labels_for_revision_quality(wikidata=wikidata)

    # labels = user_quitting_labels(wikidata)
    labels = user_reversion_label(wikidata)
    # instance_list = build_entries_for_learning(user_graph=wikidata, labels=labels)
    instance_graph = generate_instance_graph(wikidata, labels, limiter=BREADTH)
    instance_list = instance_graph.values()

    # for depth in [1,2]:

    HIDDEN_LAYER_SIZES = [2, 13, 9]
    INPUT_SIZES = [8, 8, 8]
    LEARNING_RATE_VECTOR = [0.05, 0.5, 0.5]
    DEPTH = 3
    OBJECTIVE_FUNCTION = "softmax_classification"
    NUMBER_OF_INSTANCES = 50000
    # lstm_stack = Multi_Layer_LSTM(DEPTH, HIDDEN_LAYER_SIZES, INPUT_SIZES)

    TEST_RANGE = 1

    results = {}
    total_prec_list = {}
    total_prec_list[0] = []
    total_prec_list[1] = []

    total_recall_list = {}
    total_recall_list[0] = []
    total_recall_list[1] = []

    total_f1_list = {}
    total_f1_list[0] = []
    total_f1_list[1] = []

    total_avg_recall_list = []


    for iter in range(TEST_RANGE):
        t1 = time.clock()
        lstm_stack = MultiLSTM(max_depth=DEPTH,
                               hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                               input_sizes=INPUT_SIZES,
                               instance_graph=instance_graph)
        # random.seed(500)
        random.shuffle(instance_list)

        training_set_dist = 0.60
        training_set_size = int(training_set_dist * len(instance_list))
        training_set = instance_list[0:training_set_size]
        # get labels proportion
        label_count = 0.0
        for i in training_set:
            if i.get_label() == 1.0:
                label_count += 1.0

        label_proportion = label_count / training_set_size
        print "Label proportion: ", label_proportion
        test_set = instance_list[training_set_size:len(instance_list)]

        lstm_stack.train_model_force_balance(training_set, no_of_instances=NUMBER_OF_INSTANCES, max_depth=DEPTH - 1,
                                             objective_function=OBJECTIVE_FUNCTION,
                                             learning_rate_vector=LEARNING_RATE_VECTOR)
        t2 = time.clock()
        precision_dict, recall_dict, recall_list, all_labels = lstm_stack.test_model_simple(test_set, max_depth=DEPTH - 1)

        t3 = time.clock()

        results_file = os.path.join(os.getcwd(), 'results', WIKINAME, 'results_breadth_%d_depth_%d_instances_%d.json' % (BREADTH, DEPTH, NUMBER_OF_INSTANCES))

        print "Training completed in %r"%(t2-t1)
        if os.path.isfile(results_file):
            with open(results_file, 'rb') as inp:
                results = (Serializable.loads(inp.read()))
        else:
            results = {}

        for label in all_labels:
            label = str(label)
            # total_prec_list[label].append(precision_dict[label])
            # total_recall_list[label].append(recall_dict[label])
            # total_avg_recall_list.append(np.mean(recall_list))
            # total_f1_list[label].append(_f1(precision_dict[label], recall_dict[label]))
            for keyname in ['prec','rec','f1']:
                if not results.has_key(keyname):
                    results[keyname] = {}
                if not results[keyname].has_key(label):
                    results[keyname][label] = []
            if not results.has_key('avg_rec'):
                results['avg_rec'] = []

            results['prec'][label].append(precision_dict[int(label)])
            results['rec'][label].append(recall_dict[int(label)])
            results['f1'][label].append(_f1(precision_dict[int(label)], recall_dict[int(label)]))
        results['avg_rec'].append(np.mean(recall_list))

        with open(results_file, 'wb') as outp:
            outp.write(Serializable.dumps(results))

#     results = dict(prec = total_prec_list, rec = total_recall_list, f1=total_f1_list, avg_rec=total_avg_recall_list)
#
# with open('results_depth_%d.json'%(DEPTH), 'wb') as outp:
#     outp.write(Serializable.dumps(results))

    # print "Depth was: %d"%(DEPTH)



