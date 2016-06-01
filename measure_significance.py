import numpy as np
from multi_layer_lstm.statistical_significance import statistical_significance
from json_plus import Serializable
import os

def significance_among(list1, list2, level):
    a = np.array(list1)
    b = np.array(list2)
    p_value, z_left_tail = statistical_significance(a, b, level=level)


if __name__ == "__main__":

    WIKINAME = 'astwiki'
    BREADTH = 7

    DEPTH = 1
    results_file = os.path.join(os.getcwd(), 'results', WIKINAME, 'results_breadth_%d_depth_%d.json' % (BREADTH, DEPTH))

    with open(results_file, 'rb') as inp:
        r1 = Serializable.loads(inp.read())

    DEPTH = 2
    results_file = os.path.join(os.getcwd(), 'results', WIKINAME, 'results_breadth_%d_depth_%d.json' % (BREADTH, DEPTH))
    with open(results_file, 'rb') as inp:
        r2 = Serializable.loads(inp.read())

    f1_label1_d1 = r1['f1']['0']
    f1_label1_d2 = r2['f1']['0']
    p, z = significance_among(f1_label1_d1, f1_label1_d2, level=0.05)
    print p, z

    # for k in r1.keys():
    #     if type(r1[k])!='list':
    #         for en in ['0','1']:
    #             l1 = r1[k][en]
    #             l2 = r2[k][en]
    #             p, z = significance_among(l1, l2, level=0.95)
    #             print p, z
    #     else:
    #         l1 = r1[k]
    #         l2 = r2[k]
    #         p, z = significance_among(l1, l2, level=0.95)
    #         print p,z

