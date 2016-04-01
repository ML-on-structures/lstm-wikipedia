#!/usr/bin/python

# This file contains the code for the computation of chunk
# differences.  The main function is chunk_match, which compares two
# sets of chunks.  Other functions are text_to_chunk, which makes a
# text into an initial chunk.
#
# Luca de Alfaro, March 2006

import heapq
import unittest

# The constants used to describe the edit distance,
# Format:
# (MOVE, from, to, length)
# (INSERT, _, to, length)
# (DELETE, from, _, length)
MOVE = 0
INSERT = 1
DELETE = 2

# This is the maximum number of matches that can be tolerated for a word pair.  Anything beyond this, and the word pair is not used to
MAX_MATCHES = 50
# How many words in an entry of the hash table for Greedy.  Valid values: 2, 3, 4
greedy_hash_words = 2

TICHY_HASH_WORDS = 3

##################################################################
# These are the main edit distance, if you want to know nothing of
# what goes on inside.
def compute_edit_list(s1, s2):
    """Computes the edist list from s1 to s2."""
    idx = make_index_tichy(s2)
    return edit_diff_tichy(s1, s2, idx)

def fast_compute_edit_list(s1, s2):
    """Fast version of the above, disregarding initial prefixes.
    It returns:
    - Initial Move, if any.
    - Intermediate edit list.
    - Final Move, if any.
    Keeping the initial and final moves separate is often convenient.
    """
    l1, l2 = len(s1), len(s2)
    m = min(l1, l2)
    j = 0
    while j < m and s1[j] == s2[j]:
        j += 1
    initial_move = None if j == 0 else (MOVE, 0, 0, j)
    k = 0
    while k < m - j and s1[l1 - k - 1] == s2[l2 - k - 1]:
        k += 1
    final_move = None if k == 0 else (MOVE, l1 - k, l2 - k, k)
    idx = make_index_tichy(s2[j: l2 - k])
    intermediate, added_words = edit_diff_tichy(s1[j: l1 - k], s2[j: l2 - k], idx)

    new_intermediate = [(m, i1 + j, i2 + j, l) for (m, i1, i2, l) in intermediate]
    return initial_move, new_intermediate, final_move, added_words


################################################################
#
# Quality of a match
#
# This function defines the quality of a match, which depends on how
# long the match is, and on how long the displacement is.
def quality (l, i1, len1, i2, len2):
    # l: length
    # i1, i2: starting position of match
    # len1, len2: lengths of the two revisions
    return  (1.0 * l) / min (len1, len2) - 0.3 * abs (i1 / (1.0 * len1) - i2 / (1.0 * len2))


# Makes index for greedy, of segments of 2 words.
def make_index2 (words):
    index = {}
    # the -1 in the following loop is because a match has length
    # at least 2.
    for i in xrange (len (words) - 1):
        if (words[i], words[i+1]) in index:
            (index [(words[i], words[i+1])]).append(i)
        else:
            index [(words[i], words[i+1])] = [i]
    # Now chops index entries with too many matches.
    todel = []
    for (wp, l) in index.iteritems ():
        if len(l) > MAX_MATCHES:
            todel.append (wp)
    for wp in todel:
        del index [wp]
    return index


# Makes index for greedy, of segments of 3 words.
def make_index3 (words):
    index = {}
    # the -2 in the following loop is because a match has length
    # at least 3.
    for i in xrange (len (words) - 2):
        if (words[i], words[i+1], words[i+2]) in index:
            (index [(words[i], words[i+1], words[i+2])]).append(i)
        else:
            index [(words[i], words[i+1], words[i+2])] = [i]
    # Now chops index entries with too many matches.
    todel = []
    for (wp, l) in index.iteritems ():
        if len(l) > MAX_MATCHES:
            todel.append (wp)
    for wp in todel:
        del index [wp]
    return index

# Makes index for greedy, of segments of 4 words.
def make_index4 (words):
    index = {}
    # the -2 in the following loop is because a match has length
    # at least 3.
    for i in xrange (len (words) - 3):
        if (words[i], words[i+1], words[i+2], words[i+3]) in index:
            (index [(words[i], words[i+1], words[i+2], words[i+3])]).append(i)
        else:
            index [(words[i], words[i+1], words[i+2], words[i+3])] = [i]
    # Now chops index entries with too many matches.
    todel = []
    for (wp, l) in index.iteritems ():
        if len(l) > MAX_MATCHES:
            todel.append (wp)
    for wp in todel:
        del index [wp]
    return index

# Makes index for greedy
def make_index (words):
    if greedy_hash_words == 2:
        return (make_index2 (words))
    elif greedy_hash_words == 3:
        return (make_index3 (words))
    else:
        return (make_index4 (words))

def make_index_tichy(words):
    index = {}
    for i in xrange (len (words) - TICHY_HASH_WORDS + 1):
        ws = tuple(words[i : i + TICHY_HASH_WORDS])
        if ws in index:
            (index [ws]).append(i)
        else:
            index [ws] = [i]
    # Now chops index entries with too many matches.
    todel = []
    for (wp, l) in index.iteritems ():
        if len(l) > MAX_MATCHES:
            todel.append (wp)
    for wp in todel:
        del index [wp]
    return index

#################################################################
#
# Edit difference.
#
# This function takes as input three lists of words:
# 1) The list of words of revision A.
# 2) The list of words of revision B.
# 3) The index for revision B, created with make_index.
#
# Based on this, the function returns the list of edit commands
# used to go from revision A to revision B.
#
# The function uses greedy text matching, which tries to match longest
# matches first.
# Note: matching twice the same old text IS NOT allowed.

def edit_diff_greedy(words1, words2, index2):

    # Heap to store all the matches
    matches = []

    # Now goes over words1, finding all the matches.
    len1 = len (words1)
    len2 = len (words2)
    # To avoid putting too much in the heap
    prev_matches = []
    for i1 in xrange (len1 - greedy_hash_words + 1):
        # Looks at the candidate on the other side
        if greedy_hash_words == 2:
            ws = (words1 [i1], words1 [i1 + 1])
        elif greedy_hash_words == 3:
            ws = (words1 [i1], words1 [i1 + 1], words1 [i1 + 2])
        else:
            ws = (words1 [i1], words1 [i1 + 1], words1 [i1 + 2], words1 [i1 + 3])
        if ws in index2:
            # These are the new matches
            new_matches = index2 [ws]
            for i2 in new_matches:
                # If i2 - 1 was in prev_matches, then it does not count, it is a shorter
                # match than one we have already seen.
                if i2 - 1 not in prev_matches:
                    # Looks at all matches on the other side, which start at i2
                    l = 2
                    while (i1 + l < len1 and
                           i2 + l < len2 and
                           words1 [i1 + l] == words2 [i2 + l]):
                        l += 1
                    # Ok, this match starts at words1 [i1] and words2
                    # [i2], and is of length l.
                    # Computes the quality of the match
                    q = quality (l, i1, len1, i2, len2)
                    matches.append ((10.0 - q, (l, i1, i2)))
            # The matches of this iteration are the prev_matches of the next
            prev_matches = new_matches
        else:
            # The matches of this iteration (none) are the prev_matches of the next
            prev_matches = []

    # Ok, at this point, all the matches are in the heap.
    # Marks on both sides the chunks as not yet matches.
    matched1 = [0] * len(words1)
    matched2 = [0] * len(words2)
    match_id = 0
    diff = []
    added_words = []
    # Orders the matches heap
    heapq.heapify (matches)
    # Now pulls out the matches out of the priority heap, best
    # match first.
    while matches != []:
        (q, (l, i1, i2)) = heapq.heappop (matches)
        # Checks whether it has been already matched. As we pull them
        # out longest first, we can just check the endpoints.
        if not (matched1 [i1] or matched2 [i2]):
            # Lower end not mached.
            if not (matched1 [i1 + l - 1] or matched2 [i2 + l - 1]):
                # Upper end not matched.
                # Adds the match to the set12 ...
                diff.append( (MOVE, i1, i2, l) )
                # ... and marks it matched
                match_id += 1
                for j in xrange (0, l):
                    matched1 [i1 + j] = match_id
                    matched2 [i2 + j] = match_id
            else:
                # The upper endpoint, but not the lower, is matched.
                # Figures out the maximal residual match, and puts it back into
                # the priority queue.
                j = l - 2 # we know l-1 does not work
                while (matched1 [i1 + j] or matched2 [i2 + j]): j -= 1
                l = j + 1
                if l >= greedy_hash_words:
                    q = quality (l, i1, len1, i2, len2)
                    heapq.heappush (matches, (10.0 - q, (l, i1, i2)))
        else:
            # Lower end is matched already.
            if not (matched1 [i1 + l - 1] or matched2 [i2 + l - 1]):
                # The upper end is not matched (and the lower end is).
                # Inserts the submatch in the table.
                j = 1 # we know that at j=0 they are matched already.
                while (matched1 [i1 + j] or matched2 [i2 + j]): j += 1
                l = l - j
                if l >= greedy_hash_words:
                    q = quality (l, i1 + j, len1, i2 + j, len2)
                    heapq.heappush (matches, (10.0 - q, (l, i1 + j, i2 + j)))
            else:
                # Both upper and lower end are matched already.  We need to look
                # if there is an unmatched island in the middle.
                # There cannot be an island if we are in the same match.
                if matched1 [i1] != matched1 [i1 + l - 1] and matched2 [i2] != matched2 [i2 + l - 1]:
                    j = 1
                    while (j < l - 1 and (matched1 [i1 + j] or matched2 [i2 + j])): j += 1
                    if j < l - 1:
                        # Beginning at j, there is some portion that is not matched yet.
                        k = j + 1
                        while (k < l - 1 and not (matched1 [i1 + k] or matched2 [i2 + k])): k += 1
                        # Ok, the match ends at position k - 1 (i.e., k is
                        # the first one out of the match).
                        l = k - j
                        if l >= greedy_hash_words:
                            q = quality (l, i1 + j, len1, i2 + j, len2)
                            heapq.heappush (matches, (10.0 - q, (l, i1 + j, i2 + j)))

    # Great, at this point, we simply need to find the sets of
    # unmatched 1 and 2 chunks.  First find unmatched from set1.
    # in_string tells me whether I am in the middle of an unmatched string.
    in_string = False
    for i1, m1 in enumerate(matched1):
        if not m1 and not in_string:
            # This is the start of an unmatched chunk.
            unm_start = i1
            in_string = True
        if m1 and in_string:
            # This is the end of an unmatched chunk.
            # Appends it only if the length is > 0
            if i1 > unm_start:
                diff.append( (DELETE, unm_start, unm_start, i1 - unm_start) )
            in_string = False
    # Takes care of last portion
    if in_string and len1 > unm_start :
        diff.append( (DELETE, unm_start, unm_start, len1 - unm_start) )
    # Same thing for set2
    # in_string tells me whether I am in the middle of an unmatched string.
    in_string = False
    for i2, m2 in enumerate (matched2):
        if not m2 and not in_string:
            # This is the start of an unmatched chunk.
            unm_start = i2
            in_string = True
        if m2 and in_string:
            # This is the end of an unmatched chunk.
            # Appends it only if the length is > 0
            if i2 > unm_start:
                diff.append( (INSERT, unm_start, unm_start, i2 - unm_start) )
                added_words.append(words2[unm_start:i2])
            in_string = False
    # Takes care of last portion
    if in_string and len2 > unm_start:
        diff.append( (INSERT, unm_start, unm_start, len2 - unm_start) )
        added_words.append(words2[unm_start:len2])

    return diff, added_words


# public
# Computes chunks from an initial version of a page.
def text_to_chunk (label_data, text):
    """Converts some text, with label, to a list of chunks containing
    only one chunk."""
    chunk = text.split()
    chunk_label = (0, len(chunk), label_data )
    return (chunk, [chunk_label], [], [])

# public
# Prepares some text into a revision (single element) chunk.
def revision_prepare (text):
    return text.split ()


#################################################################
#
# Edit difference.
#
# This function takes as input three lists of words:
# 1) The list of words of revision A.
# 2) The list of words of revision B.
# 3) The index of revision B
#
# Based on this, the function returns the list of edit commands
# used to go from revision A to revision B.
#
# The function uses Tichy text matching, in case it turns out to be faster.
# Note: matching twice the same old text IS NOT allowed.

def edit_diff_tichy (w1, w2, idx2):

    # I want an uniform interface with Greedy...
    words1 = w2
    words2 = w1
    index1 = idx2
    len1 = len (words1)
    len2 = len (words2)

    # Marks on both sides the chunks as not yet matches.
    # +1: start match. -1: end match.
    matched1 = [False] * len(words1)

    # Looks for longest matches of strings in chunk2, starting from start2.
    diff = [] # list of differences
    added_words = []  # list of added words
    deleted_words = []  # list of deleted words
    i2 = 0
    # This is the first unmatched element on side 2.
    first_unmatched_2 = 0
    # Remember, chunks have length at least 2
    while i2 < len2 - TICHY_HASH_WORDS + 1:
        max_quality = None
        ws = tuple(words2 [i2 : i2 + TICHY_HASH_WORDS])
        if ws in index1:
            for i1 in index1 [ws]:
                # Now that we found a partial match, determine the length
                if not any(matched1 [i1 : i1 + TICHY_HASH_WORDS]):
                    l = TICHY_HASH_WORDS
                    while (i1 + l < len1 and
                           i2 + l < len2 and
                           (not matched1 [i1 + l]) and
                           words1 [i1 + l] == words2 [i2 + l]):
                        l += 1
                    # Ok, we have a match.
                    # Is it the best quality one?
                    q = quality (l, i1, len1, i2, len2)
                    if max_quality == None or q > max_quality:
                        max_quality = q
                        best_match = (l, i1)

            # Appends the best match, and marks it as matched
            if max_quality != None:
                #  First, inserts the unmatched part of words2, if needed.
                if i2 > first_unmatched_2:
                    diff.append ( (DELETE, first_unmatched_2, first_unmatched_2, i2 - first_unmatched_2) )
                    deleted_words.append(words1[first_unmatched_2:i2])
                # Then, inserts the match
                (l_m, i1_m) = best_match
                diff.append ( (MOVE, i2, i1_m, l_m) )
                for j in xrange (l_m):
                    matched1 [i1_m + j] = True
                # Jumps to the end of the match
                i2 += l_m
                first_unmatched_2 = i2
            else:
                # All matches are already matched
                i2 += 1
        # No matches
        else:
            i2 += 1

    # Takes care of the final unmatched part.
    if len2 > first_unmatched_2:
        # Issue a Block Insert for the final unmatched part
        diff.append( (DELETE, first_unmatched_2, first_unmatched_2, len2 - first_unmatched_2) )
        deleted_words.append(words1[first_unmatched_2:len2])

    # Ok! Now all that remains to do is to produce the list of unmatched portions of word1.
    in_string = False
    for i1, m1 in enumerate(matched1):
        if not m1 and not in_string:
            # This is the start of an unmatched chunk.
            unm_start = i1
            in_string = True
        if m1 and in_string:
            # This is the end of an unmatched chunk.
            diff.append( (INSERT, unm_start, unm_start, i1 - unm_start) )
            added_words.append(words2[unm_start:i1])
            in_string = False
    # Takes care of last portion
    if in_string:
        diff.append( (INSERT, unm_start, unm_start, len1 - unm_start) )
        added_words.append(words2[unm_start:len1])

    # All done!
    return diff, added_words, deleted_words


def print_edit_diff(diff_cmds):
    for (cmd, i1, i2, l) in diff_cmds:
        if cmd == MOVE:
            print "MOVE   : %d -> %d, L= %d" % (i1, i2, l)
        if cmd == DELETE:
            print "DELETE : %d, L: %d" % (i1, l)
        if cmd == INSERT:
            print "INSERT : %d, L: %d" % (i1, l)

def human_edit_diff(diff_cmds):
    s = ''
    for (cmd, i1, i2, l) in diff_cmds:
        if cmd == MOVE:
            s += "(MOVE   : %d -> %d, L= %d) " % (i1, i2, l)
        if cmd == DELETE:
            s += "(DELETE : %d, L: %d) " % (i1, l)
        if cmd == INSERT:
            s += "(INSERT : %d, L: %d) " % (i1, l)
    return s

def test_tichy(s1, s2):
    l1 = s1.split()
    l2 = s2.split()
    idx = make_index_tichy(l2)
    return edit_diff_tichy(l1, l2, idx)

def test_greedy(s1, s2):
    l1 = s1.split()
    l2 = s2.split()
    idx = make_index2(l2)
    return edit_diff_greedy(l1, l2, idx)

class TestDiffsGreedy(unittest.TestCase):

    def test1(self):
        s1 = "la capra canta contenta sotto la collina sopra la panca la capra campa"
        s2 = "sotto la panca la capra crepa e la capra canta"
        r = test_greedy(s1, s2)
        print s1
        print s2
        print_edit_diff(r)
        print r
        self.assertEqual(r, [(0, 8, 1, 4), (0, 0, 7, 3), (2, 3, 3, 5),
                             (2, 12, 12, 1), (1, 0, 0, 1), (1, 5, 5, 2)])

    def test2(self):
        s1 = "la capra canta contenta sotto la collina sopra la panca la capra campa"
        s2 = "sotto la panca la capra crepa e la capra canta"
        print s1
        print s2
        r = test_greedy(s1, s2)
        print_edit_diff(r)
        print r
        self.assertEqual(r, [(0, 8, 1, 4), (0, 0, 7, 3), (2, 3, 3, 5),
                             (2, 12, 12, 1), (1, 0, 0, 1), (1, 5, 5, 2)])

    def test3(self):
        s1 = "nel bel mezzo del cammin di nostra vita mi trovai in una selva oscura che la diritta via era smarrita"
        s2 = "nel frammezzo del cammin di nostra esistenza mi trovai nel bel mezzo di una selva oscura dove la via era smarrita e non mi trovai nel cammin di casa nostra"
        print s1
        print s2
        r = test_greedy(s1, s2)
        print_edit_diff(r)
        print r
        self.assertEqual(r, [(0, 3, 2, 4), (0, 11, 13, 3), (0, 17, 18, 3),
                             (0, 0, 9, 3), (0, 8, 7, 2), (2, 7, 7, 1), (2, 10, 10, 1),
                             (2, 14, 14, 3), (1, 0, 0, 2), (1, 6, 6, 1), (1, 12, 12, 1),
                             (1, 16, 16, 2), (1, 21, 21, 9)])

    def test4(self):
        s1 = "A me piace la pasta alla matriciana ma zenza parmigiano sopra"
        s2 = "A me piace la pasta al pomodoro con il parmigiano sopra"
        r = test_greedy(s1, s2)
        x, w, y = fast_compute_edit_list(s1.split(), s2.split())
        self.assertItemsEqual(r, [x] + w + [y])

class TestDiffsTichy(unittest.TestCase):

    def test1(self):
        s1 = "la capra canta contenta sotto la collina sopra la panca la capra campa"
        s2 = "sotto la panca la capra crepa e la capra canta"
        r = test_tichy(s1, s2)
        print_edit_diff(r)
        self.assertEqual(r, [(0, 0, 3, 2), (2, 2, 2, 2), (0, 4, 0, 2), (2, 6, 6, 4), (0, 10, 7, 2),
                             (2, 12, 12, 1), (1, 2, 2, 1), (1, 5, 5, 2), (1, 9, 9, 1)])

    def test2(self):
        s1 = "la capra canta contenta sotto la collina sopra la panca la capra campa"
        s2 = "sotto la panca la capra crepa e la capra canta"
        r = test_tichy(s1, s2)
        print_edit_diff(r)
        self.assertEqual(r, [(0, 0, 3, 2), (2, 2, 2, 2), (0, 4, 0, 2), (2, 6, 6, 4), (0, 10, 7, 2),
                             (2, 12, 12, 1), (1, 2, 2, 1), (1, 5, 5, 2), (1, 9, 9, 1)])

    def test3(self):
        s1 = "nel bel mezzo del cammin di nostra vita mi trovai in una selva oscura che la diritta via era smarrita"
        s2 = "nel frammezzo del cammin di nostra esistenza mi trovai nel bel mezzo di una selva oscura dove la via era smarrita e non mi trovai nel cammin di casa nostra"
        r = test_tichy(s1, s2)
        print_edit_diff(r)
        self.assertEqual(r, [(0, 0, 9, 3), (0, 3, 2, 4), (2, 7, 7, 1), (0, 8, 7, 2), (2, 10, 10, 1),
                             (0, 11, 13, 3), (2, 14, 14, 3), (0, 17, 18, 3), (1, 0, 0, 2), (1, 6, 6, 1),
                             (1, 12, 12, 1), (1, 16, 16, 2), (1, 21, 21, 9)])

    def test4(self):
        s1 = "a me piacciono le banane"
        s2 = "a me piacciono me piacciono le banane"
        r = test_tichy(s1, s2)
        x, w, y = fast_compute_edit_list(s1.split(), s2.split())
        self.assertItemsEqual(r, [x] + w + [y])

    def test5(self):
        s1 = "A me piace la pasta alla matriciana ma zenza parmigiano sopra"
        s2 = "A me piace la pasta al pomodoro con il parmigiano sopra"
        r = test_tichy(s1, s2)
        x, w, y = fast_compute_edit_list(s1.split(), s2.split())
        self.assertItemsEqual(r, [x] + w + [y])

    def test6(self):
        s1 = "a b c d h a b c d e a b c d a b c d"
        s2 = "a b c d a b c d"
        x, w, y = fast_compute_edit_list(s1.split(), s2.split())
        self.assertItemsEqual([(0, 0, 0, 4), (2, 4, 4, 10), (0, 14, 4, 4)], [x] + w + [y])

if __name__ == '__main__':
    unittest.main()

