import os
import gzip
from pprint import pprint

import numpy as np
import parse

##############
## Test Mode
import re

user_graph = {}


def add_to_graph(file_content):
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
    print "User Graph now"
    pprint(user_graph)
    print "\n-------------\n"
    lines = file_content.splitlines()
    for line in lines:
        pattern = "EditInc (?P<time>\d+) PageId: (?P<pageid>\d+) Delta: (?P<delta_char>\d+).(?P<delta_mant>\d+) rev0: (?P<rev0>\d+) uid0: (?P<uid0>\d+) uname0: (?P<uname0>\S+) rev1: (?P<rev1>\d+) uid1: (?P<uid1>\d+) uname1: (?P<uname1>\S+) rev2: (?P<rev2>\d+) uid2: (?P<uid2>\d+) uname2: (?P<uname2>\S+) d01: (?P<d01_char>\d+).(?P<d01_mant>\d+) d02: (?P<d02_char>\d+).(?P<d02_mant>\d+) d12: (?P<d12_char>\d+).(?P<d12_mant>\d+) dp2: (?P<dp2_char>\d+).(?P<dp2_mant>\d+) n01: (?P<n01>\d+) n12: (?P<n12>\d+) t01: (?P<t01_char>\d+).(?P<t01_mant>\d+) t12: (?P<t12_char>\d+).(?P<t12_mant>\d+)"
        m = re.match(pattern=pattern, string=line)
        if m is not None:
            # print line
            # pprint(m.groupdict())
            vals = m.groupdict()
            k = vals['uname0']
            if user_graph.has_key(k):
                if user_graph[k].has_key(vals['uname1']):
                    user_graph[k][vals['uname1']].append({})
                else:
                    user_graph[k][vals['uname1']] = []
                    user_graph[k][vals['uname1']].append({})
            else:
                user_graph[k] = {}
                user_graph[k][vals['uname1']] = []
                user_graph[k][vals['uname1']].append({})


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


if __name__ == "__main__":
    base_dir = "/home/rakshit/Research/ML/wikipedia_lstm/data/wikitrust"

    get_files(base_dir)

    pprint(user_graph)