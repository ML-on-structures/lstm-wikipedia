import json
import os
import random
import re
from pprint import pprint

import numpy as np
import time

from datetime import datetime
from pydal import DAL, Field

import chdiff
from editdist import edit_distance
from serializer import json_to_data, data_to_json
from wikipedia import WikiFetch

# Some constants to be used in DataAccess operations
DATE_PATTERN = "%Y-%m-%dT%H:%M:%SZ"  # "%Y-%b%a, %d %b %Y %H:%M:%S %z"
START_TIME = "2012-01-01T01:01:01Z"
ALPHA = 0.5
SECS_IN_HR = 3600
HRS_IN_WEEK = 168





def _normalize_inputs(features, gen_values, last_two=False):
    """
    Normalize the set of features to be in the range [0,1]
    The NN uses this range and the inputs must be kept within
    this range.

    Each input is normalized in its own way.
    Features:
            time_prev_user
            time_prev_page
            time_prev_user_page
            chars_added
            chars_removed
            spread
            position_in_page
            time_in_day
            day_of_week
            rev_comment_length
            upper_lower_ratio
            digit_total_ratio


    :param features: Features of entry
    :param gen_values: General values to be used for normalization
    :param last_two: Boolean to control whether last two (time_to_next, quality) values should be touched
    :return: normalized array for each row
    """
    norm_list = []
    # Time values - 0,1,2
    # Clip them
    # print features
    norm_list = [np.clip(i, 0.0, 1.0) for i in features[:3]]

    # Char measurement -3,4
    # Z-score
    char_add_val = 1.0 * (features[3] - gen_values.get(('char_add_mean'))) / gen_values.get('char_add_std')
    char_add_val = np.clip(char_add_val, -3.0, 3.0)
    char_add_val = 1.0 * (char_add_val + 3) / 6.0
    norm_list.append(char_add_val)

    char_rem_val = 1.0 * (features[4] - gen_values.get(('char_rem_mean'))) / gen_values.get('char_rem_std')
    char_rem_val = np.clip(char_rem_val, -3.0, 3.0)
    char_rem_val = 1.0 * (char_rem_val + 3) / 6.0
    norm_list.append(char_rem_val)

    # Unchanged - 5,6
    norm_list.append(features[5])
    norm_list.append(features[6])

    # Time in day - 7
    # sin(2*pi*h/24)
    new_val = np.sin(2 * np.pi * features[7] / 24)
    # Put in [0,1] range
    norm_list.append(1.0 * (new_val + 1.0) / 2.0)

    # Day of week - 8
    # Divide by 7
    norm_list.append(1.0 * features[8] / 7.0)

    # Rev comment Lenght - 9
    # Z-score
    comm_len_val = 1.0 * (features[9] - gen_values.get(('com_len_mean'))) / gen_values.get('com_len_std')
    comm_len_val = np.clip(comm_len_val, -3.0, 3)
    comm_len_val = 1.0 * (comm_len_val + 3) / 6.0
    norm_list.append(comm_len_val)

    # Upper lower ratio - 10
    # Divide by 2 and clip between [0,1]
    ul_original = features[10]
    ul_new = np.clip(1.0 * ul_original / 2.0, 0, 1)
    norm_list.append(ul_new)

    # Digit total ratio - 11
    # Unchanged
    norm_list.append(features[11])

    if last_two:
        # Time to next -12
        # Unchanged
        norm_list.append(features[12])

        # Quality -12
        # Scale
        q_new = 1.0 * (features[13] + 1.0) / 2.0
        norm_list.append(q_new)

    return np.array(norm_list)


def _operate_on_contributions(c, username):
    """
    Operate on the contributions sent for a user.

    :param contributions:
    :return:
    """
    w = WikiFetch
    user_revision = w.fetch_revisions_for_page(pageid=c['pageid'],
                                               start_rev=c['revid'],
                                               chunk_size=1, )

    prev_revision = w.fetch_revisions_for_page(pageid=c['pageid'],
                                               start_rev=c['revid'],
                                               chunk_size=1,
                                               direction="older",  # Gets older revisions
                                               exclude=username)  # excludes same user

    next_revisions = w.fetch_revisions_for_page(pageid=c['pageid'],
                                                start_rev=c['revid'],
                                                chunk_size=10,
                                                direction="newer",  # Gets later revisions
                                                exclude=username)

    return user_revision, prev_revision, next_revisions


def _get_distances(r1, r2, distance_only=False):
    """
    Get distance between two revisions r1 and r2.
    Uses chdiff.py and editdist.py

    :param r1:
    :param r2:
    :param distance_only:
    :return:
    """
    results = {}  # Dict for results

    l1 = r1.split()  # tokenize first revision
    l2 = r2.split()  # tokenize second revision

    # Compute distance using chdiff
    x, w, y, added_words, deleted_words = chdiff.fast_compute_edit_list(l1, l2)
    minlen = min(len(l1), len(l2))

    # Get list of distances from chdiff
    # The head and tail of list have to be attached to the main w
    a = ([x] if x is not None else []) + w + ([y] if y is not None else [])
    distance = edit_distance(a, minlen)

    # Return the distance if only distance is needed.
    if distance_only:
        return distance

    # Get addition and deletion lists
    addition_list = [i[3] for i in a if i[0] == chdiff.INSERT]
    deletion_list = [i[3] for i in a if i[0] == chdiff.DELETE]

    # Position list for any action on page
    action_position_list = [i[1] for i in a]

    # Separate addition and deletion position lists
    # ---Currently not being used
    # add_position_list = [i[2] for i in a if i[0] == chdiff.INSERT]
    # del_position_list = [i[1] for i in a if i[0] == chdiff.DELETE]

    # Set char added and char removed in the result dict
    results['chars_added'] = sum(addition_list)
    results['chars_removed'] = sum(deletion_list)

    # Measure spread in the page
    # Spread is std dev of locations normalized by size of page
    results['spread'] = np.std(action_position_list) / (1.0 * len(l1))

    # Measure position in the page
    # Position in page is weighted average of all change locations
    results['position_in_page'] = np.clip(np.average(action_position_list, weights=[i[3] for i in a]) / (1.0 * len(l1)),
                                          0.0,
                                          1.0)

    # Generate added text from the changes
    # Added text
    added_text = ""
    for word in added_words:
        for t in word:
            added_text = added_text + unicode(t) + " "
        added_text += "; "
    # Words added in the revision are stored as a long string of
    # each word chunk separated by semicolon
    results['revision_added_text'] = added_text[:5000]

    # Generate deleted text from the changes
    # Added text
    deleted_text = ""
    for word in deleted_words:
        for t in word:
            deleted_text = deleted_text + unicode(t) + " "
        deleted_text += "; "
    # Words added in the revision are stored as a long string of
    # each word chunk separated by semicolon
    results['revision_deleted_text'] = deleted_text[:5000]

    # Measure ratios for added text data

    # Create auxilary lists to be used in ratio calculation
    uc = sum([sum([1 for i in x if unicode(i).isupper()]) for x in added_text])
    lc = sum([sum([1 for i in x if unicode(i).islower()]) for x in added_text])
    digit = sum([sum([1 for i in x if unicode(i).isdigit()]) for x in added_text])

    # Uppercase/ Lowercase Ratio
    results['upper_lower_ratio'] = (1.0 * uc) / (1.0 * lc) if lc else 1.0

    # Digit/ Total Ratio
    results['digit_total_ratio'] = (1.0 * digit) / (1.0 * len(added_text)) if len(added_text) else 0.0

    # Return the dict holding these character features
    return results


def _get_previous_by_user_on_page(username, page, revision):
    """
    Get the previous revision by user with username
    on the given page. Using the revision provided.

    :param username: Username of user
    :param page: Page to which revision belongs (pageid)
    :param revision: Revision being used currently (revid)
    :return:
    """
    if revision is None:
        return 0

    w = WikiFetch()
    prev_revision = w.fetch_revisions_for_page(pageid=page,
                                               start_rev=revision,
                                               chunk_size=2,
                                               direction="older",
                                               include=username)
    if len(prev_revision) < 2:
        return 0

    t_val = datetime.strptime(prev_revision[1].get('timestamp'), DATE_PATTERN)

    return t_val


def _measure_revision_quality(curr, prev, foll, next_count):
    """
    Measure the quality of Current revision c,  using Previous revision p
    and upto next n revisions denoted as k_j:j \in {1...n)
    and upto next_count of the foll revisions

    Quality is calculated by first getting distance of c with p, calling it pc.
    Then for each k_j:
        Get the distance of k_j from c and p.

        So now we have three distances pc, pk_j and ck_j.

        Quality of c based on k_j is calculated as the
        difference between distance pk_j and ck_j
        divided by distnace pc
        Means,
        difference of how much change exists between k_j and previous,
        and between k_j and current. This difference is a representative
        of what was retained from c till k_j
        Dividing this by the distance of current from previous basically
        divides the retention by the amount of change actually performed.
        This value should be between -1 and 1 where
        -1 means that all of the change was reverted back and
        +1 means that all of the change was retained

    Final quality is a weighted average of quality with respect to n next revisions
    weighted by their sequence distance from current (eg. 1, 2, ...n)

    :param curr: Current revision object (A list with a single dict object)
    :param prev: Previous revision object (A list with a single dict object)
    :param foll: 10 next revisions (A list with a multiple dict objects)
    :param next_count: Number of revisions to use from foll list
    :return: Measured quality (float)
    """
    # Initializing empty lists for Q_j's and weights
    qjs = []  # Quality measure from jth following revision
    wjs = []  # weight assigned to jth revision's quality

    # Get content of current and previous revisions
    content_curr = curr.get('*', '')
    content_prev = prev[0].get('*', '')

    # Measure with each next revision
    if len(foll):
        for pos, v in enumerate(foll[:next_count]):
            # Get content of this following revision
            content_foll = v.get('*', '')

            # Measure three distances
            # Distance of current from previous
            dist_pc = _get_distances(content_curr, content_prev, distance_only=True)
            # Distance of following from current
            dist_cf = _get_distances(content_curr, content_foll, distance_only=True)
            # Distance of following from previous
            dist_pf = _get_distances(content_prev, content_foll, distance_only=True)

            # Now calculate the q_j for this following revision
            q_j = 1.0 * ((dist_pf - dist_cf) / (1.0 * dist_pc)) if dist_pc else 0.0

            # Provide a weight to this quality which is a function of
            # the revision's sequence distnace from current revision
            w_j = np.exp((-1) * ALPHA * (pos + 1))

            # Add q_j and weight to a list for averaging later
            qjs.append(q_j)
            wjs.append(w_j)

        # Clip all qjs to stay within [-1,1] range
        qjs = np.clip(qjs, -1.0, 1.0)

    # Get quality as weighted average of all if they exist.
    # Otherwise return 2.0 to make it an outlier in future prediction usage
    quality = np.average(qjs, weights=wjs) if len(qjs) else 2.0

    return quality


class DataAccess:
    """

    """

    def __init__(self):
        self.db = DAL('sqlite://storage4.sqlite', folder="data")

        self.authors = self.db.define_table('authors',
                                            Field('username', unique=True),
                                            # , requires=[IS_NOT_IN_DB(db, 'authors.username')]),
                                            Field('userid', 'integer'),
                                            Field('contributions', 'integer'),
                                            Field('user_since', 'datetime'),
                                            Field('cleaned', 'boolean', default=False),
                                            Field('completed', 'boolean', default=False),
                                            migrate=False)

        self.revisions = self.db.define_table('revisions_extended',
                                              Field('revid', 'integer', unique=True),
                                              Field('pageid', 'integer'),
                                              Field('rev_timestamp', 'datetime'),
                                              Field('rev_comment'),
                                              Field('username'),
                                              Field('userid', 'integer'),
                                              Field('rev_content', 'text'),
                                              Field('rev_size'),
                                              Field('time_prev_user', 'double'),
                                              Field('time_prev_page', 'double'),
                                              Field('time_prev_user_page', 'double'),
                                              Field('time_next_page', 'double'),
                                              Field('chars_added', 'integer'),
                                              Field('chars_removed', 'integer'),
                                              Field('spread', 'double'),
                                              Field('position_in_page', 'double'),
                                              Field('time_in_day', 'double'),
                                              Field('day_of_week', 'integer'),
                                              Field('rev_comment_length', 'integer'),
                                              Field('upper_lower_ratio', 'double'),
                                              Field('digit_total_ratio', 'double'),
                                              Field('revision_added_text', 'text'),
                                              Field('processed', 'boolean', default=False),
                                              Field('quality', 'double'),
                                              migrate=False
                                              )

        self.revisions2 = self.db.define_table('revisions_restore',
                                              Field('revid', 'integer', unique=True),
                                              Field('pageid', 'integer'),
                                              Field('parentid', 'integer'),
                                              Field('rev_timestamp', 'datetime'),
                                              Field('rev_comment'),
                                              Field('username'),
                                              Field('userid', 'integer'),
                                              Field('rev_content', 'text'),
                                              Field('rev_size'),
                                              Field('time_prev_user', 'double'),
                                              Field('time_prev_page', 'double'),
                                              Field('time_prev_user_page', 'double'),
                                              Field('time_next_page', 'double'),
                                              Field('current_rev_length', 'integer'),
                                              Field('parent_rev_length', 'integer'),
                                              Field('chars_added', 'integer'),
                                              Field('chars_removed', 'integer'),
                                              Field('spread', 'double'),
                                              Field('position_in_page', 'double'),
                                              Field('time_in_day', 'double'),
                                              Field('day_of_week', 'integer'),
                                              Field('rev_comment_length', 'integer'),
                                              Field('upper_lower_ratio', 'double'),
                                              Field('digit_total_ratio', 'double'),
                                              Field('revision_added_text', 'text'),
                                              Field('revision_deleted_text', 'text'),
                                              Field('processed', 'boolean', default=False),
                                              Field('q4', 'double'),
                                              Field('q6', 'double'),
                                              Field('q10', 'double'),
                                              migrate=True
                                              )

        self.db.commit()

    def _get_main_normalization_values(self):

        """
        Contact revisions_extended table and measure required
        std and avg values per item
        These are DB level values to be used on individual normalizations
        :return: Dicrtionary of means and stds.
        """
        ret_dict = {}

        data = self.db().select(self.revisions.ALL)
        com_lengths = [i.rev_comment_length for i in data]
        ret_dict['com_len_mean'] = np.mean(com_lengths)
        ret_dict['com_len_std'] = np.std(com_lengths)

        chars_add = [i.chars_added for i in data]
        ret_dict['char_add_mean'] = np.mean(chars_add)
        ret_dict['char_add_std'] = np.std(chars_add)

        chars_rem = [i.chars_removed for i in data]
        ret_dict['char_rem_mean'] = np.mean(chars_rem)
        ret_dict['char_rem_std'] = np.std(chars_rem)

        return ret_dict

    def get_list_from_previous(self):
        """

        :return:
        """
        user_list = self.db().select(self.revisions.username, distinct=True)

        return user_list

    def collect_contributions(self, lim_start=1, lim_end=1000, user_list=None):
        """
        Get upto 50 first contributions of a user.
        Users are already pre-fetched into the DB.
        For each user in the limit, get the revisions 1-n with max n being 50.

        While getting each revision, perform operations to extract features out of the revisions.
        Features to fetch are as follows:
            Time features:
                Time from user's previous edit
                Time from previous contribution on revision's page
                Time from user's previous edit on this page

            Character features:
                Characters added
                Characters removed
                Spread within page
                Position in page
                Added text
                UpperCase/LowerCase ratio
                Digit/Total ratio

            Action features:
                Revision comment length
                Time in day (UTC)
                Day of week

            Future features:
                Time to next revision on page
                Quality of revision

        This function updates the revision entries into 'revisions_extended' table
        To identify an author with revisions collected, the boolean value 'completed' in table 'authors'
        is set to True if the user's edits are stored.

        :return: Returns total count of revisions fetched
        """

        # Object for WikiFetch class (to communicate with Wikipedia)
        w = WikiFetch()

        # Rveision count for total sum
        total_rev_count = 0
        # Get users from the table. Controlled by 'cleaned' and 'completed'
        q = (
                self.authors.user_since > START_TIME) & (
                self.authors.contributions > 3) & (
                #self.authors.cleaned == True) & (
                self.authors.completed == False
            )
        if not user_list:
            users = self.db(q).select(limitby=(lim_start, lim_end))
        else:
            users = user_list

        print "Length of users: %r"%(len(users))
        # Get revisions for each user from Wikipedia
        for i in users:

            # This piece of code is kept in a try-except block because
            # it is very long running and sometimes due to unknown
            # errors it tends to consider some DB fields with Nan
            # values even when the code works totally fine.
            #
            # There are places where due to certain errors in values
            # we stop considering a particular user and move on to next.
            # Since we have a really large number of users, we can avoid
            # such issues by not using the users.
            try:

                if not user_list:

                    # Remove if it is a bot
                    if re.search("bot",i['username'],re.IGNORECASE):
                        self.db(self.authors.id == i['id']).delete()
                        print("{} deleted".format(i['username']))
                        continue

                    # Basic user values available in 'authors' table
                    username = i['username']

                else:
                    username = i['username']

                # Initialize a boolean to control completion of record
                completed = True

                # Get the contributions for this user.
                # Getting up to first 50
                # Using the method written in wikipedia.py
                # This call returns a list of contributions by this author
                # with each entry of that list being a dict
                contributions = w.get_user_contributions(username=username,
                                                         cont_limit=50, )
                print "Contributions by user (%r) are: %r"%(username, len(contributions))

                # Don't operate if it has less than 3 revisions
                if len(contributions) < 3:
                    continue

                # Iterate over these contibutions and get their features
                for c, v in enumerate(contributions):
                    # Each contribution must be inserted into the database now with all supporting values
                    # So at this stage, we merge basic fields of the revision with
                    # extracted features using a set of methods.

                    # To control previous revision in future calculations
                    c_before = contributions[c - 1] if c else None

                    # Wikipedia PageID of the revision under consideration
                    pageid = v.get('pageid')

                    # curr is this revision
                    # prev is previous revision by another author
                    # following is list of next (upto 10) revisions by different authors
                    curr, prev, following = _operate_on_contributions(v, username=username)

                    # Some checks since web data can often lead to unknown errors. (eg 500 from Wikipedia server)
                    # Current revision and previous revision by another author on page are
                    # important for quality measurements. So make sure they exist
                    if not curr or not prev:
                        completed = False
                        break

                    # Get individual entry from list curr
                    curr = curr[0]

                    # Basic features of current revision
                    t_curr = datetime.strptime(curr.get('timestamp'), DATE_PATTERN)
                    content_curr = curr.get('*', '')
                    parent_curr = curr.get('parentid', None)

                    # Get parent revision of current using parentID
                    # Features of parent revision are required in order to
                    # measure distance and char update values of this revision
                    parent_rev = w.fetch_revisions_for_page(pageid=pageid,
                                                            start_rev=parent_curr,
                                                            chunk_size=1, )
                    # Do not use the revision if parent_rev can't be fetched.
                    if not parent_rev:
                        completed = False
                        break

                    # Basic features of parent revision
                    parent_rev = parent_rev[0]
                    content_parent = parent_rev.get('*', '')
                    t_prev_page = datetime.strptime(parent_rev.get('timestamp'), DATE_PATTERN)

                    # Get next revision on page for future time feature
                    # The call will include current revision also
                    next_rev = w.fetch_revisions_for_page(pageid=pageid,
                                                          start_rev=curr['revid'],
                                                          chunk_size=2,
                                                          direction="newer", )

                    # Check if next revision was retrieved or not
                    # If not then do not continue with this
                    if len(next_rev) < 2:
                        completed = False
                        break

                    # To get time feature from next revision on this page
                    next_rev = next_rev[1]
                    t_next_page = datetime.strptime(next_rev.get('timestamp'), DATE_PATTERN)

                    # Get distances from parent revision
                    # Here we get character features by comparing current revision with parent revision
                    feature_dict = _get_distances(content_curr, content_parent)
                    feature_dict['current_rev_length'] = len(content_curr)
                    feature_dict['parent_rev_length'] = len(content_parent)

                    # Now add action features to this set of character features
                    feature_dict['rev_comment_length'] = len(curr.get('comment', ''))
                    feature_dict['time_in_day'] = t_curr.hour
                    feature_dict['day_of_week'] = t_curr.weekday()

                    # Getting time features now by using current, previous and next tinmes.

                    # Time from previous revision on page
                    feature_dict['time_prev_page'] = (t_curr - t_prev_page).total_seconds()

                    # Time to next on page
                    feature_dict['time_next_page'] = (t_next_page - t_curr).total_seconds()

                    # Time from previous revision by user
                    # The call will include current revision also
                    contribution_before = w.get_user_contributions(username=username,
                                                                   cont_limit=2,
                                                                   start_time=curr.get('timestamp'),
                                                                   direction="older")

                    t_user_prev = datetime.strptime(contribution_before[1].get('timestamp'), DATE_PATTERN) if len(
                        contribution_before) > 1 else 0

                    feature_dict['time_prev_user'] = (t_curr - t_user_prev).total_seconds() if t_user_prev else 0.0

                    # Time from previous revision by user on this page
                    t_user_page_prev = _get_previous_by_user_on_page(username=username,
                                                                     page=pageid,
                                                                     revision=curr.get('revid', None))

                    feature_dict['time_prev_user_page'] = (t_curr - t_user_page_prev).total_seconds() if t_user_page_prev else 0.0


                    # Fill in remaining entries of revision dict
                    # to be placed in the DB. These include values from
                    # result dict obtained by calling Wikimedia API
                    feature_dict['revid'] = curr.get('revid')
                    feature_dict['pageid'] = pageid
                    feature_dict['parentid'] = parent_curr
                    feature_dict['username'] = username
                    feature_dict['rev_timestamp'] = t_curr  # To get it as datetime type
                    feature_dict['userid'] = curr['userid']
                    feature_dict['rev_content'] = content_curr
                    feature_dict['rev_comment'] = curr.get('comment', '')
                    feature_dict['rev_size'] = curr['size']

                    # Now we need to calculate the quality of this revision by
                    # using a revision prior to it from a different author and
                    # using next 10 revisions by different authors
                    # Measure quality for current revision
                    feature_dict['q4'] = _measure_revision_quality(curr=curr, prev=prev, foll=following, next_count=4)
                    feature_dict['q6'] = _measure_revision_quality(curr=curr, prev=prev, foll=following, next_count=6)
                    feature_dict['q10'] = _measure_revision_quality(curr=curr, prev=prev, foll=following, next_count=10)

                    # Can print and look at the dict if needed
                    # print("===== DICT printing====")
                    # pprint(feature_dict)
                    # print("===== DICT printed====")

                    # Push revision into the DB

                    insert_return = self.revisions2.update_or_insert(self.revisions2.revid == curr.get('revid'),
                                                       **feature_dict)
                    # Commit at this point to ensure it stays in DB even if something else crashes
                    self.db.commit()

                    # Update total revision count
                    total_rev_count += 1

                # Use completed boolean to update authors table flags
                if completed:
                    updates_user = self.db(self.authors.username == username).update(completed=True)
                    print "User updated to complete"
                else:
                    updates_user = self.db(self.authors.username == username).update(cleaned=False)

                self.db.commit()

            except:
                # Basically if the revision fetch process crashed for any unknown reason,
                # just continue the loop and move to next user.
                continue

        return total_rev_count

    def load_fresh_from_db(self, store=True, limit_users=50):
        """
        This function should only be used if users and revisions
        already exist in the database. Without that step, it would mostly create empty structures.

        Load revisions from the Database.
        Normalize their values and produce training and test dicts.
    
        :param store: Bool to control whether to store data in files or not. Default is True
        :param limit_users: None or int to control no. of users to be used. Default is None which means all users
        :return: Return two dicts training and test
        """

        NUMBER_OF_FEATURES = 14
        LIMITER = -1

        t_start = time.clock()

        # Get users from the DB where revisions are available
        users = self.db(self.authors.completed == True).select()

        # Initialize empty dicts
        training_dict = {}
        test_dict = {}

        # Basic normalization values from full Revisions set
        print "Getting base normalization values after %r seconds" % (time.clock() - t_start)
        main_normalizer_values = self._get_main_normalization_values()

        # Start getting revisions for each user
        print "Starting the loop after %r seconds" % (time.clock() - t_start)
        for i in users:
            revisions = self.db(self.revisions.username == i.username).select()

            # Check to remove small sized entries
            if len(revisions) < 3:
                continue

            # Set matrix dimension for the user
            # Dimension X is for the number of revisions from 1-(n-1)
            dim_x = len(revisions[:LIMITER])
            # Dimension y is for the number of features being used per revision
            dim_y = NUMBER_OF_FEATURES
            # Initialize an empty matrix for User's 1-(n-1) revision features
            mat_ur = np.zeros((dim_x, dim_y))

            # Getting features from the DB per revision
            for ctr, c in enumerate(revisions[:LIMITER]):
                l = [c.time_prev_user,
                     c.time_prev_page,
                     c.time_prev_user_page,
                     c.chars_added,
                     c.chars_removed,
                     c.spread,
                     c.position_in_page,
                     c.time_in_day,
                     c.day_of_week,
                     c.rev_comment_length,
                     c.upper_lower_ratio,
                     c.digit_total_ratio,
                     c.time_next_page,
                     c.quality]

                # Get normalized values for features
                l_normalized = _normalize_inputs(l, main_normalizer_values, last_two=True)
                mat_ur[ctr] = l_normalized

            # Create last row features (nth revision)
            last_row = revisions[LIMITER]

            # Set size of this row to be features without two (time to next revision and the quality)
            vect_y_features = [last_row.time_prev_user,
                               last_row.time_prev_page,
                               last_row.time_prev_user_page,
                               last_row.chars_added,
                               last_row.chars_removed,
                               last_row.spread,
                               last_row.position_in_page,
                               last_row.time_in_day,
                               last_row.day_of_week,
                               last_row.rev_comment_length,
                               last_row.upper_lower_ratio,
                               last_row.digit_total_ratio]

            # Normalize and get back in the form of a numpy array
            vect_y_features = _normalize_inputs(vect_y_features, main_normalizer_values)

            # Quality of last (nth) revision. At this point it is not normalized and is in range [-1,1]
            # The quality when used for loss measurement needs to be normalized (by scaling it into [0,1] range)
            y_quality = last_row.quality

            # Randomly distribute to training and test
            # Structure of each entry in the dicts is like:
            #       {author: (Matrix of past revisions vs. features,
            #                   Features of the nth revision,
            #                   Quality (label) of the nth revisoin)}
            if random.random() > 0.20:
                training_dict[i.username] = (mat_ur, vect_y_features, y_quality)
            else:
                test_dict[i.username] = (mat_ur, vect_y_features, y_quality)

            # Store:
            if store:
                # Store the entries into json files
                training_file = os.path.join(os.getcwd(), 'data', 'trainig_data.json')
                test_file = os.path.join(os.getcwd(), 'data', 'test_data.json')
                try:
                    # Try to backup last data
                    os.rename(training_file, training_file + ".bak")
                    os.rename(test_file, test_file + ".bak")
                except:
                    pass

                with open(training_file, 'wb') as output:
                    json.dump(data_to_json(training_dict), output)

                with open(test_file, 'wb') as output:
                    json.dump(data_to_json(test_dict), output)

        return training_dict, test_dict

    def get_missing_data(self):
        """
        Iterate over existing data and fill the missing fields.
        Get parent revision id and content for each revision.
        Get removed text from revision
        """
        w = WikiFetch

        revs = self.db().select(self.revisions.ALL)

        for i in revs[:10]:
            revid = i.revid
            pageid = i.pageid

            # Get revision for revid

            curr = w.fetch_revisions_for_page(pageid=pageid,
                                                   start_rev=revid,
                                                   chunk_size=1, )

            if not curr:
                completed = False
                continue

            # Get individual entry from list curr
            curr = curr[0]

            # Extract parent ID
            t_curr = datetime.strptime(curr.get('timestamp'), DATE_PATTERN)
            content_curr = curr.get('*', '')
            parent_curr = curr.get('parentid', None)

            # Get parent revision of current using parentID
            # Features of parent revision are required in order to
            # measure distance and char update values of this revision
            parent_rev = w.fetch_revisions_for_page(pageid=pageid,
                                                    start_rev=parent_curr,
                                                    chunk_size=1, )

            # Do not use the revision if parent_rev can't be fetched.
            if not parent_rev:
                completed = False
                continue

            # Basic features of parent revision
            parent_rev = parent_rev[0]
            content_parent = parent_rev.get('*', '')
            t_prev_page = datetime.strptime(parent_rev.get('timestamp'), DATE_PATTERN)

            # Collect removed text
            feature_dict = _get_distances(content_curr, content_parent)

            i.update_record(revision_deleted_text=feature_dict['revision_deleted_text'], parentid=parent_curr)

        print len(revs)

if __name__ == "__main__":
    db = DataAccess()
    # users = db.db(db.authors.completed == True).select()
    # #print users
    # print len(users)
    #
    # for i in users[50:80]:
    #     revs = db.db(db.revisions.username == i.username).count()
    #     print revs
    #     print "______________________"

    # Collect data
    # db.collect_contributions(lim_start=100, lim_end=3200)

    # Previous list round
    # print "Getting list of previous users"
    # user_list = db.get_list_from_previous()
    # print "List now available with %r users"%(len(user_list))
    # db.collect_contributions(lim_start=1, lim_end=1000, user_list=user_list)

    # Get missing data per entry. Then improve rest
    db.get_missing_data()