"""
This file is for the functions that will analyze a graph generated
for a wiki using user_graph.

The graph can be for any language wiki.

Functions in this file should build labels from a graph,
build required LSTM models and run them.
"""
import json
import os
from pprint import pprint
from datetime import datetime

import numpy as np


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
    return (timestamp1-timestamp2).total_seconds()

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
    return v/3600/24


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

    delta_dict = {k: _timediff(maxt,v) for k, v in timestamps.iteritems()}

    pprint(delta_dict)

    l2_list = delta_dict.values()

    min_l2 = min(l2_list)
    max_l2 = max(l2_list)
    avg_l2 = np.average(l2_list)


    print min_l2, max_l2, avg_l2
    histo =  np.histogram(l2_list)

    print histo
    for i in histo[1]:
        print _days(i)

    quit_labels = {k:0 if _days(v)>40 else 1 for k,v in delta_dict.iteritems()}

    print np.histogram(quit_labels.values())

if __name__ == "__main__":
    # Get the data for concerned wiki

    graph_file = os.path.join(os.getcwd(), 'results', 'user_graph_test.json')

    with open(graph_file, 'rb') as inp:
        wikidata = json.load(inp)

    contrib_file = os.path.join(os.getcwd(), 'results', 'user_contrib_test.json')

    with open(contrib_file, 'rb') as inp:
        user_contribs = json.load(inp)

    # Build labels

    build_labels_for_user_quitting(user_contribs=user_contribs)
    # build_labels_for_revision_quality(wikidata=wikidata)
