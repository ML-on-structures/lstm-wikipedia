import json
import os
import gzip
import random
from pprint import pprint

from datetime import datetime
import numpy as np

##############
## Test Mode
import re

from db import DataAccess

# user_graph = {}
global NONECTR
NONECTR = 0

def add_to_graph(file_content, user_graph):
    """

    From WikiTrust:
          EditInc 1113671446 PageId: 7954 Delta:    2.50 rev0: 10550535 uid0: 55767 uname0: "R. fiend" rev1: 11805828 uid1: 0 uname1: "210.49.80.154" rev2: 12543256 uid2: 231414 uname2: "Jack39" d01:   16.00 d02:  611.67 d12:  613.17  dp2: 302.00 n01: 6 n12: 3 t01: 3429205 t12: 1543097
        This line is used to compute an increment (positive or negative) to the reputation of the author of a revision,
        according to how much the edits are still present at a later revision.
        EditInc ;; fixed string.
        1113671446 ;; timestamp at which the edit was made.
        PageId: 7954 ;; the page id is 1.
            Delta: 2.50 ;; how much was the edit distance between this revision and the previous one
        rev0: 10550535 ;; there are the revision id, user id, and user name of the revision that is the reference point in the past.
        uid0: 55767 ;;
            uname0: "R. fiend" ;;
        rev1: 11805828 ;;this is the revision whose change is being judged.
        uid1: 0 ;; '0' means anonymous
            uname1: "210.49.80.154" ;; for anonymous users, the name is the IP
        rev2: 12543256 ;; this is the judge revision, userid, and username, in the future of rev1.
        uid2: 231414 ;;
            uname2: "Jack39" ;;
        d01: 16.00 ;; The distance between rev0 and rev1
        d02: 611.67 ;; The distance between rev1 and rev2
        d12: 613.17 ;; The distance between rev0 and rev2
            dp2: 302.00 ;; the distance between the revision before rev1, and rev2
            n01: 6 ;; The number of revisions between rev0 and rev1, including rev1.
            n12: 3 ;; The number of revisions between rev1 and rev2, including rev2.
            t01: 3429205 ;; The elapsed time between rev0 and rev1.
            t12: 1543097 ;; The elapsed time between rev1 and rev2.

    :param file_content:
    :type file_content: str
    :return:
    :rtype:
    """
    # print "User Graph now"
    # pprint(user_graph)
    # print "\n-------------\n"
    global NONECTR
    db = DataAccess()
    lines = file_content.splitlines()

    for line in lines:
        # pattern = "EditInc (?P<time>\d+) PageId: (?P<pageid>\d+) Delta: (?P<delta_char>\d+).(?P<delta_mant>\d+) rev0: (?P<rev0>\d+) uid0: (?P<uid0>\d+) uname0: (?P<uname0>\S+) rev1: (?P<rev1>\d+) uid1: (?P<uid1>\d+) uname1: (?P<uname1>\S+) rev2: (?P<rev2>\d+) uid2: (?P<uid2>\d+) uname2: (?P<uname2>\S+) d01: (?P<d01_char>\d+).(?P<d01_mant>\d+) d02: (?P<d02_char>\d+).(?P<d02_mant>\d+) d12: (?P<d12_char>\d+).(?P<d12_mant>\d+) dp2: (?P<dp2_char>\d+).(?P<dp2_mant>\d+) n01: (?P<n01>\d+) n12: (?P<n12>\d+) t01: (?P<t01_char>\d+).(?P<t01_mant>\d+) t12: (?P<t12_char>\d+).(?P<t12_mant>\d+)"
        #
        # pattern = """EditInc(\s*)(?P<time>\d+)(\s*)PageId:(\s*)(?P<pageid>\d+)(\s*)Delta:(\s*)(?P<delta_char>\d+).(?P<delta_mant>\d+)(\s*)rev0:(\s*)(?P<rev0>\d+)(\s*)uid0:(\s*)(?P<uid0>\d+)(\s*)uname0:(\s*)"(?P<uname0>\S+)"(\s*)rev1:(\s*)(?P<rev1>\d+)(\s*)uid1:(\s*)(?P<uid1>\d+)(\s*)uname1:(\s*)"(?P<uname1>\S+)"(\s*)rev2:(\s*)(?P<rev2>\d+)(\s*)uid2:(\s*)(?P<uid2>\d+)(\s*)uname2:(\s*)"(?P<uname2>\S+)"(\s*)d01:(\s*)(?P<d01_char>\d+).(?P<d01_mant>\d+)(\s*)d02:(\s*)(?P<d02_char>\d+).(?P<d02_mant>\d+)(\s*)d12:(\s*)(?P<d12_char>\d+).(?P<d12_mant>\d+)(\s*)dp2:(\s*)(?P<dp2_char>\d+).(?P<dp2_mant>\d+)(\s*)n01:(\s*)(?P<n01>\d+)(\s*)n12:(\s*)(?P<n12>\d+)(\s*)t01:(\s*)(?P<t01_char>\d+).(?P<t01_mant>\d+)(\s*)t12:(\s*)(?P<t12_char>\d+).(?P<t12_mant>\d+)"""
        # m = re.search(pattern=pattern, string=line)
        #

        line_broken = line.split('|')
        if line_broken[0]=="EditInc":
            line_dict = {i.split(':')[0]:i.split(':')[1] for i in line_broken[2:]}
            line_dict['timestamp'] = line_broken[1]

            # Add entry to DB
            db.graph_edge.insert(**line_dict)
            db.ug_db.commit()

            keys_to_remove =['uid0','uid1','uid2','rev0','rev1','rev2','uname1','uname2']

            node_dict = line_dict.copy()
            for key in keys_to_remove:
                node_dict.pop(key)

            u0 = line_dict['uname0']
            u1 = line_dict['uname1']
            u2 = line_dict['uname2']

            if not user_graph.has_key(u1):
                user_graph[u1] = {}

            if not user_graph[u1].has_key(u2):
                user_graph[u1][u2] = []

            user_graph[u1][u2].append(node_dict)

        else:
            # print "Reached into None"
            # print line
            # print "\n\n"
            if "EditInc" in line:
                print line
            NONECTR+=1


def get_files(base_dir):
    """
    Get all the files in nested directories under base_dir

    :param base_dir: Absolute path of the base directory
    :type base_dir: str
    :return:
    :rtype:
    """


    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            for r2, d2, f2 in os.walk(os.path.join(root, dir)):
                for file in f2:
                    with gzip.open(os.path.join(r2, file), 'rb') as input:
                        file_content = input.read()
                        add_to_graph(file_content)


def build_graph():
    """
    The graph of users with each user being a node and edges being the respresentative of relation among them.

    The graph looks like this:

        user_graph = {
            'user_1': {
                'user_2': [
                    {
                        'p1': 2.0,
                        'p2': 12.131,
                        'p3': "fcfrw"
                    },
                    {
                        'p1': -2.0,
                        'p2': 12322.131,
                        'p3': "aafac"
                    }
                ],

                'user_3': [
                    {
                        'p1': 2.0,
                        'p2': 12.131,
                        'p3': "fcfrw"
                    },
                    {
                        'p1': -2.0,
                        'p2': 12322.131,
                        'p3': "aafac"
                    }
                ]
            },

            'user_2': {
                'user_3': [
                    {
                        'p1': 2.0,
                        'p2': 12.131,
                        'p3': "fcfrw"
                    },
                    {
                        'p1': -2.0,
                        'p2': 12322.131,
                        'p3': "aafac"
                    }
                ],

                'user_1': [
                    {
                        'p1': 2.0,
                        'p2': 12.131,
                        'p3': "fcfrw"
                    },
                    {
                        'p1': -2.0,
                        'p2': 12322.131,
                        'p3': "aafac"
                    }
                ]
            }
        }

    :return:
    :rtype:
    """


def get_split_files(base_dir, user_graph):
    """
    Get all the files in nested directories under base_dir

    :param base_dir: Absolute path of the base directory
    :type base_dir: str
    :return:
    :rtype:
    """

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            with gzip.open(os.path.join(root, file), 'rb') as input:
                file_content = input.read()
                add_to_graph(file_content, user_graph)


if __name__ == "__main__":
    base_dir = "/home/rakshit/Research/ML/wikipedia_lstm/data/wikitrust/001"

    user_graph_file = os.path.join(os.getcwd(), 'results', 'user_graph_test.json')

    if os.path.isfile(user_graph_file):
        with open(user_graph_file, 'rb+') as inp:
            user_graph = json.load(inp)
    else:
        user_graph = {}
    # get_files(base_dir)

    get_split_files(base_dir, user_graph)

    with open(user_graph_file, 'wb+') as output:
        json.dump(user_graph,output)
    # pprint(user_graph)
    print "Users in graph", len(user_graph.keys())
    print "None counter", NONECTR

    for i in random.sample(user_graph.keys(),10):
        pprint(user_graph[i])
        print "--------"