#!/usr/bin/python2.4

# This file contains the code for computing the edit distance
# between two pieces of text. It relies on chdiff.py to compute the list
# of edit commands.
#
# Luca de Alfaro, May 2006

import operator

import chdiff


def edit_distance (diff, length):
    """diff is the list of edits; length is the length of the page
    (one of the two pages). length is used for renormalization."""

    # Computes the edit distance given a list of diffs.
    distance_ins = 0.0 # This is the total insertions
    distance_del = 0.0 # This is the total deletions
    mov_list = [] # List of moves. 
    for (op, i1, i2, l) in diff:
        if op == chdiff.INSERT:
            distance_ins += l
        elif op == chdiff.DELETE:
            distance_del += l 
        else:
            # Makes a list of the moves. 
            mov_list.append ( (op, i1, i2, l) )
    # Ok, now sorts the list of moves wrt i1.
    mov_list.sort (key = operator.itemgetter (1))
    # Now we must re-sort the list wrt i2, performing only adjacent
    # swaps, and summing the cost of all swaps.
    # Assuming matches are somewhat in order, the code uses
    # an up-down bounded bubblesort function, which sorts only
    # between lower_b and upper_b.
    lower_b = 0
    upper_b = len (mov_list) - 1
    distance_move = 0.0 # This is the distance due to moves
    while upper_b > lower_b + 1: 
        # First, we go up.
        top_change = 0
        # print "range:", range (lower_b, upper_b) # debug 
        for i in xrange (lower_b, upper_b):
            (op, i1, i2, l)         = mov_list [i]
            (op_1, i1_1, i2_1, l_1) = mov_list [i + 1]
            if (i2 > i2_1): 
                # Exchanges
                (mov_list [i], mov_list [i + 1]) = (mov_list [i + 1], mov_list [i])
                # Records the distance_move
                # print "move:", i, mov_list [i], mov_list [i + 1] # debug 
                distance_move += (l * l_1) / (1.0 * length)
                # Records that it has been shifted
                top_change = i
        # Now we go down
        upper_b = top_change 
        bottom_change = upper_b
        # print "range:", range (upper_b, lower_b, -1) # debug 
        for i in xrange (upper_b, lower_b, -1):
            (op, i1, i2, l)         = mov_list [i]
            (op_1, i1_1, i2_1, l_1) = mov_list [i - 1]
            if (i2 < i2_1): 
                # Exchanges
                (mov_list [i], mov_list [i - 1]) = (mov_list [i - 1], mov_list [i])
                # Records the distance_move
                # print "move:", i, mov_list [i], mov_list [i - 1] # debug 
                distance_move += (l * l_1) / (1.0 * length)
                # Records that it has been shifted
                bottom_change = i
        lower_b = bottom_change


    # The total distance is:
    # distance_ins + distance_del - min(distance_ins + distance_del)
    # + 0.5 * min(distance_ins + distance_del)
    # + distance_move
    dmax = max (distance_ins, distance_del)
    dmin = min (distance_ins, distance_del)
    return (dmax - 0.5 * dmin + distance_move)

# debug 
if False: 
    print edit_distance ([(chdiff.MOVE, 2, 20, 2),
                          (chdiff.MOVE, 3, 7, 4),
                          (chdiff.MOVE, 4, 9, 3),
                          (chdiff.MOVE, 5, 17, 2),
                          (chdiff.MOVE, 6, 14, 3),
                          (chdiff.MOVE, 7, 40, 4),
                          (chdiff.MOVE, 8, 2, 1),
                          (chdiff.MOVE, 9, 25, 2),
                          (chdiff.MOVE, 10, 30, 3),
                          (chdiff.INSERT, 2, 0, 4),
                          (chdiff.INSERT, 2, 0, 3),
                          (chdiff.INSERT, 2, 0, 2),
                          (chdiff.DELETE, 2, 0, 1),
                          (chdiff.DELETE, 2, 0, 2),
                          (chdiff.DELETE, 2, 0, 3)], 40)
    
