import json
import os
import gzip
import random
import uuid
from pprint import pprint
from json_plus import Serializable
from datetime import datetime
import numpy as np

##############
## Test Mode
import re

from db import DataAccess

WIKINAME = 'rmywiki'
USER_INDEX = os.path.join(os.getcwd(), 'results', WIKINAME, 'user_index.json')

FILE_MODE = False
# user_graph = {}
global NONECTR
NONECTR = 0


def add_to_graph(file_content):  # , user_graph, user_contribs):
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
        if line_broken[0] == "EditInc":
            line_dict = {i.split(':')[0]: i.split(':')[1] for i in line_broken[2:]}
            line_dict['timestamp'] = line_broken[1]

            # Add entry to DB
            entry_id = db.graph_edge.insert(**line_dict)
            db.ug_db.commit()

            if np.random.randn() > 0.99:
                print "Entry ID for DB: %r" % (entry_id)

                # keys_to_remove =['uid0','uid1','uid2','rev0','rev1','rev2','uname1','uname2']
                #
                # node_dict = line_dict.copy()
                # for key in keys_to_remove:
                #     node_dict.pop(key)
                #
                # u0 = line_dict['uname0']
                # u1 = line_dict['uname1']
                # u2 = line_dict['uname2']
                #
                # if not user_graph.has_key(u1):
                #     user_graph[u1] = {}
                #
                # if not user_graph[u1].has_key(u2):
                #     user_graph[u1][u2] = []
                #
                # user_graph[u1][u2].append(node_dict)
                #
                # # User contribution record
                #
                # if not user_contribs.has_key(u1):
                #     user_contribs[u1] = []
                #
                # user_contribs[u1].append(line_dict)


        else:
            # print "Reached into None"
            # print line
            # print "\n\n"
            if "EditInc" in line:
                print line
            NONECTR += 1


def _get_index_entry(user):
    if os.path.isfile(USER_INDEX):
        with open(USER_INDEX, 'rb') as inp:
            index_of_users = json.load(inp)
    else:
        index_of_users = {}

    return index_of_users.get(user, _set_into_index(user))


def _set_into_index(user):
    if os.path.isfile(USER_INDEX):
        with open(USER_INDEX, 'rb') as inp:
            index_of_users = json.load(inp)
    else:
        index_of_users = {}

    index_of_users[user] = str(uuid.uuid5(uuid.NAMESPACE_OID, user))

    with open(USER_INDEX, 'wb') as outp:
        json.dump(index_of_users, outp)

    return index_of_users.get(user)


def _get_file_name(user, suffix='dict'):
    index_entry_for_user = _get_index_entry(user)

    user_file = "%s_%s.json" % (index_entry_for_user, suffix)
    return os.path.join(os.getcwd(), 'results', WIKINAME, user_file)


def _get_dict_for_user(user):
    """
    Get the file containing revision entries for the user "user"
    If the file is present, load its json which is effectively a dict

    Otherwise create the file
    :param user:
    :type user:
    :return:
    :rtype:
    """
    if FILE_MODE:
        filename = _get_file_name(user)
        if os.path.isfile(filename):
            with open(filename, 'rb') as inp:
                user_dict = Serializable.loads(json.load(inp))
        else:
            user_dict = {user: {}}


    else:
        filename = os.path.join(os.getcwd(),'results',WIKINAME,'user_graph.json')
        with open(filename, 'rb') as inp:
            user_dict = Serializable.loads(json.load(inp))

        if not user_dict.has_key(user):
            user_dict[user] = {}

    return user_dict[user]


def _update_dict_for_user(user, dict_for_user):
    """
    Write the dict into user's specific file
    :param user:
    :type user:
    :return:
    :rtype:
    """
    if FILE_MODE:
        dict_to_dump = {user: dict_for_user}

        filename = _get_file_name(user)
        with open(filename, 'wb') as outp:
            json.dump(Serializable.dumps(dict_to_dump), outp)

    else:
        filename = os.path.join(os.getcwd(), 'results', WIKINAME, 'user_graph.json')
        with open(filename, 'rb') as inp:
            user_dict = Serializable.loads(json.load(inp))

        user_dict[user] = dict_for_user

        with open(filename, 'wb') as outp:
            json.dump(Serializable.dumps(user_dict),outp)




def _update_edge(user, rev, full_dict, user_graph=None):
    """
    Update the dictionary for user witha list to the revisions rev.
    The full dict consists of an item which needs to be added to the list of
    work done on tope of rev.

    :param user:
    :type user:
    :param rev:
    :type rev:
    :param full_dict:
    :type full_dict:
    :return:
    :rtype:
    """

    if user_graph is not None:

        # Update entry of this user in graph
        if not user_graph.has_key(user):
            user_graph[user] = {}

        if not user_graph[user].has_key(rev):
            user_graph[user][rev] = {}
            user_graph[user][rev]['timestamp'] = int(full_dict['timestamp']) - int(full_dict['t12'])
            user_graph[user][rev]['list'] = []
        user_graph[user][rev]['list'].append(full_dict)

        return user_graph
    else:

        user_dict = _get_dict_for_user(user)

        # rev_key = (rev, full_dict['timestamp'] - full_dict['t12'])
        rev_key = rev
        if not user_dict.has_key(rev_key):
            user_dict[rev_key] = {}
            user_dict[rev_key]['timestamp'] = int(full_dict['timestamp']) - int(full_dict['t12'])
            user_dict[rev_key]['list'] = []

        user_dict[rev_key]['list'].append(full_dict)

        _update_dict_for_user(user, user_dict)


def add_graph_content(file_content, user_graph=None):
    """

    :param file_content:
    :type file_content:
    :return:
    :rtype:
    """
    lines = file_content.splitlines()
    for line in lines:
        line_broken = line.split('|')
        if line_broken[0] == "EditInc":
            line_dict = {i.split(':')[0]: i.split(':')[1] for i in line_broken[2:]}
            line_dict['timestamp'] = line_broken[1]

            # This entry in line_dict represents judgement of uname1's rev1 revision by uname2 using rev2
            if user_graph is not None:
                _update_edge(user=line_dict['uname1'], rev=line_dict['rev1'], full_dict=line_dict, user_graph=user_graph)
            else:
                _update_edge(user=line_dict['uname1'], rev=line_dict['rev1'], full_dict=line_dict)


def get_files(base_dir):  # , user_graph, user_contribs):
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
                        # add_to_graph(file_content)
                        add_graph_content(file_content=file_content)


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


def get_split_files(base_dir):  # , user_graph):
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
                add_to_graph(file_content)


def get_user_dict(base_dir):
    """
    Get all the files in nested directories under base_dir

    :param base_dir: Absolute path of the base directory
    :type base_dir: str
    :return:
    :rtype:
    """
    user_graph = {}
    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            for r2, d2, f2 in os.walk(os.path.join(root, dir)):
                for file in f2:
                    with gzip.open(os.path.join(r2, file), 'rb') as input:
                        file_content = input.read()
                        # add_to_graph(file_content)
                        add_graph_content(file_content=file_content, user_graph=user_graph)
    return user_graph


if __name__ == "__main__":
    base_dir = os.path.join(os.getcwd(),'data','%s_pipe/stats/' % (WIKINAME))
    print base_dir
    print os.path.isdir(base_dir)

    if not os.path.isdir(os.path.join(os.getcwd(), 'results', WIKINAME)):
        os.mkdir(os.path.join(os.getcwd(), 'results', WIKINAME))

    # user_graph_file = os.path.join(os.getcwd(), 'results', 'user_graph_test.json')
    #
    # if os.path.isfile(user_graph_file):
    #     with open(user_graph_file, 'rb+') as inp:
    #         user_graph = json.load(inp)
    # else:
    #     user_graph = {}
    #
    # user_contrib_file = os.path.join(os.getcwd(), 'results', 'user_contrib_test.json')
    #
    # if os.path.isfile(user_contrib_file):
    #     with open(user_contrib_file, 'rb+') as inp:
    #         user_contribs = json.load(inp)
    # else:
    #     user_contribs = {}

    # get_files(base_dir, user_graph, user_contribs)

    # get_split_files(base_dir)
    if FILE_MODE:
        get_files(base_dir)

    else:
        user_graph = get_user_dict(base_dir)
        filename = os.path.join(os.getcwd(), 'results', WIKINAME, 'user_graph.json')
        with open(filename, 'wb') as outp:
            json.dump(Serializable.dumps(user_graph),outp)
    # with open(user_graph_file, 'wb+') as output:
    #     json.dump(user_graph, output)
    #
    # with open(user_contrib_file, 'wb+') as output:
    #     json.dump(user_contribs, output)
    #
    # # pprint(user_graph)
    # print "Users in graph", len(user_graph.keys())
    print "None counter", NONECTR

    # for i in random.sample(user_graph.keys(),10):
    #     pprint(user_graph[i])
    #     print "--------"
