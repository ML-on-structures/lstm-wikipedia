import json
import os
from pprint import pprint

from db import DataAccess
import numpy as np
import random
from datetime import datetime


DATE_PATTERN = "%Y-%m-%dT%H:%M:%SZ"  # "%Y-%b%a, %d %b %Y %H:%M:%S %z"
START_TIME = "2012-01-01T01:01:01Z"
ALPHA = 0.5
SECS_IN_HR = 3600
HRS_IN_WEEK = 168


def _get_main_normalization_values():
    """
    Contact revisions_extended table and measure required
    std and avg values per item
    These are DB level values to be used on individual normalizations
    :return: Dicrtionary of means and stds.
    """
    ret_dict = {}

    data = db.db().select(db.revisions.ALL)
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


def _normalize_inputs(features, gen_values, last_two=False):
    """
    Normalize the set of features to be in the range [0,1]
    The NN uses this range and the inputs must be kept within
    this range.

    Each input is normalized in its own way.

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

    :param features: Features of entry
    :param gen_values: General values to be used for normalization
    :param last_two: Boolean to control whether last two (time_to_next, quality) values should be touched
    :return: normalized array for each row
    """
    norm_list = []
    # Time values - 0,1,2
    # Clip them
    #print features
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
    norm_list.append(1.0*(new_val+1.0)/2.0)

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


def _load_fresh_from_db(store=True):
    """
    Load revisions from the Database.
    Normalize their values and produce training and test dicts.

    :param store: Bool to control whether to store data in files or not. Default is True
    :return: Return two dicts training and test
    """

    NUMBER_OF_FEATURES = 14
    LIMITER = -1

    db = DataAccess()
    # Get users from the DB where revisions are available
    users = db.db(db.authors.completed == True).select()

    # Initialize empty dicts
    training_dict = {}
    test_dict = {}

    print "Getting base normalization values after %r seconds"%((datetime.now()-t1).total_seconds())
    # Basic normalization values from full results set
    main_normalizer_values = _get_main_normalization_values()

    print "Starting the loop after %r seconds"%((datetime.now()-t1).total_seconds())
    for i in users[:50]:
        revisions = db.db(db.revisions.username == i.username).select()

        if len(revisions) < 3:
            continue

        dim_x = len(revisions[:LIMITER])
        dim_y = NUMBER_OF_FEATURES

        mat_ur = np.zeros((dim_x, dim_y))

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

            l_normalized = _normalize_inputs(l, main_normalizer_values, last_two=True)
            mat_ur[ctr] = l_normalized

        t_del = (revisions[LIMITER].rev_timestamp - revisions[LIMITER - 1].rev_timestamp).total_seconds() / (
            1.0 * SECS_IN_HR)

        last_row = revisions[LIMITER]
        vect_y_features = np.zeros((NUMBER_OF_FEATURES - 2))
        # Normalize
        #last_row = _normalize_inputs(last_row, main_normalizer_values)
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

        vect_y_features = _normalize_inputs(vect_y_features, main_normalizer_values)

        y_quality = last_row.quality

        if random.random() > 0.20:
            training_dict[i.username] = (mat_ur, vect_y_features, y_quality)
        else:
            test_dict[i.username] = (mat_ur, vect_y_features, y_quality)

    return training_dict, test_dict

if __name__ == "__main__":

    db = DataAccess()
    t1 = datetime.now()

    # training, test = _load_fresh_from_db()
    #
    # print len(training)
    # print len(test)
    #
    training_file = os.path.join(os.getcwd(), 'data','try_trainig_data.json')
    test_file = os.path.join(os.getcwd(), 'data','try_test_data.json')

    from json_plus import Serializable
    # training_data = Serializable.dumps(training)
    # test_data = Serializable.dumps(test)
    #
    # with open(training_file, 'wb') as output:
    #     json.dump(training_data, output)
    #
    # with open(test_file, 'wb') as output:
    #     json.dump(test_data, output)

    with open(training_file, 'rb') as input:
        tr = Serializable.loads(json.load(input))

    with open(test_file, 'rb') as input:
        tst = Serializable.loads(json.load(input))

    print len(tr)
    pprint(tr)
    print len(tst)