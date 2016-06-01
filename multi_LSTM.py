"""
Multi Layer LSTM based on mult_layer_LSTM

This version is for dictionary based structure where a dictionary with
each item of graph and its edges is defined in a way to use
them completely from the graph itself.
And then the graph is made available to this multi Layer LSTM structure which can access
the elements at any depth by picking them up from the general dict
"""
import random
from json_plus import Serializable
import multi_layer_lstm.lstm as lstm
import numpy as np


SEQUENCE_FUNCTIONS = ['none','none','none']

class MultiLSTM(Serializable):
    """
    Class to hold the multi layer LSTM model
    """

    def __init__(self,max_depth, hidden_layer_sizes, input_sizes, instance_graph):
        """

        :param max_depth: Number of LSTMs to be generated
        :type max_depth: int
        :param hidden_layer_sizes: Hidden layer per LSTM
        :type hidden_layer_sizes: list
        :param input_sizes: Input size of each LSTM
        :type input_sizes:list
        """
        self.lstm_stack = [lstm.LSTM() for l in range(max_depth)]
        for l in range(max_depth):
            self.lstm_stack[l].initialize(input_sizes[l] + (0 if l== max_depth -1 else hidden_layer_sizes[l + 1]), hidden_layer_sizes[l])
        self.hidden_layer_sizes = hidden_layer_sizes
        self.input_sizes = input_sizes
        self.instance_graph = instance_graph

    def _get_instance_node(self, link_node):
        instance_node = self.instance_graph.get(link_node[0],None)
        if instance_node is not None:
            instance_node.helper_value = {'time':link_node[1]}
        return instance_node

    def forward_instance(self, instance_node, current_depth, max_depth, sequence_function = SEQUENCE_FUNCTIONS):
        """
        Perform a complete forward run along the multi later LSTM architecture on entire depth

        :param instance_node: Instance node under concern containing its structure and children
        :type instance_node: InstanceNode
        :param current_depth: Depth defining stage of the recursive calls
        :type current_depth: int
        :param max_depth: Maximum depth to control the recursion
        :type max_depth: int
        :param sequence_function:
        :type sequence_function:
        :return:
        :rtype:
        """

        # Last case of recursion to return the identifier for this state
        if instance_node.get_number_of_children() == 0:
        # if current_depth==max_depth:
            return -100 * np.ones(self.hidden_layer_sizes[current_depth]) # no children signifier vector

        input_sequence = np.array([])
        children_sequence = instance_node.get_sequence(sequence_function=sequence_function[current_depth],
                                                       help_value=instance_node.get_help_values())


        # children_sequence = get_sequence(instance_node.get_children(), sequence_function[current_depth])
        for item in children_sequence:
            feature_vector = item.get_feature_vector()

            # If we are not at the very bottom we need to get input from LSTM at the next level
            LSTM_output_from_below = np.array([])
            if current_depth < max_depth:
                 LSTM_output_from_below = self.forward_instance(self._get_instance_node(item.get_link_node()), current_depth + 1, max_depth).reshape(self.hidden_layer_sizes[current_depth +1]) # recursive call

            # Get the full feature vector using both features of item and output from layer below
            full_feature_vector = np.concatenate((LSTM_output_from_below, feature_vector))

            # Concatenate current feature vector to input sequence for the LSTM
            input_sequence = np.concatenate((input_sequence,full_feature_vector))

        # forward the input sequence to this depth's LSTM
        # input_sequence = input_sequence.reshape(instance_node.get_number_of_children(), 1, len(full_feature_vector))
        input_sequence = input_sequence.reshape(len(children_sequence), 1, len(full_feature_vector))

        # Perform the forward operation
        _, _, Y, cache = self.lstm_stack[current_depth]._forward(input_sequence)
        instance_node.cache[current_depth] = cache
        instance_node.children_sequence = children_sequence
        return softmax(Y)

    def calculate_backward_gradients(self, instance_node, derivative, current_depth, max_depth):
        """
        Partial step for backpropagation.
        :param instance_node:
        :type instance_node:
        :param derivative:
        :type derivative:
        :param current_depth:
        :type current_depth:
        :param max_depth:
        :type max_depth:
        :return:
        :rtype:
        """
        dX, g, _, _ = self.lstm_stack[current_depth].backward_return_vector_no_update(d=derivative,
                                                                                      cache=instance_node.cache[current_depth])
        instance_node.gradient[current_depth] = g
        if current_depth == max_depth:
            return
        counter = 0
        for item in instance_node.children_sequence:
            node_for_item = self._get_instance_node(item.get_link_node())
            if node_for_item.cache.get(current_depth,None) is None:
                continue
            self.calculate_backward_gradients(node_for_item, dX[counter, :, 0:self.hidden_layer_sizes[current_depth + 1]],
                                              current_depth + 1, max_depth=max_depth)
            counter += 1

    def update_LSTM_weights(self, instance_node, current_depth, max_depth, learning_rate_vector):
        if not instance_node.gradient.get(current_depth,None) is None:
            self.lstm_stack[current_depth].WLSTM -= learning_rate_vector[current_depth] * instance_node.gradient[current_depth]
        if current_depth == max_depth:
            return
        for item in instance_node.children_sequence:
            node_for_item = self._get_instance_node(item.get_link_node())
            self.update_LSTM_weights(node_for_item, current_depth + 1, max_depth, learning_rate_vector)


    def sgd_train_multilayer(self, root, target, max_depth, objective_function, learning_rate_vector):
        # first pass the instance root one forward so that all internal LSTM states
        # get calculated and stored in "cache" field
        Y = self.forward_instance(root, current_depth=0, max_depth=max_depth)
        deriv = getDerivative(output=Y, target=target, objective=objective_function)
        self.calculate_backward_gradients(root, deriv, 0, max_depth)
        self.update_LSTM_weights(root, 0, max_depth, learning_rate_vector=learning_rate_vector)

    def train_model_force_balance(self, training_set, no_of_instances, max_depth, objective_function, learning_rate_vector):
        counter = 0
        if no_of_instances == 0:
            return
        for item in get_balanced_training_set(training_set, self.hidden_layer_sizes[0]):
            if item.get_number_of_children() == 0:
                continue
            target = np.zeros((1, self.hidden_layer_sizes[0]))
            target[0, item.get_label()] = 1.0
            self.sgd_train_multilayer(item, target, max_depth, objective_function, learning_rate_vector)
            counter += 1
            if counter % 1000 == 0:
                print "Training has gone over", counter, " instances.."
            if counter == no_of_instances:
                break

    def test_model_simple(self, test_set, max_depth):
        guesses = 0
        hits = 0
        found = {}
        missed = {}
        misclassified = {}
        for item in test_set:
            Y = self.forward_instance(item, 0, max_depth)
            if Y is None:
                continue
            # print Y
            predicted_label = Y.argmax()
            real_label = item.get_label()
            # print "Predicted label ", predicted_label, " real label", real_label
            guesses += 1
            hits += 1 if predicted_label == real_label else 0
            if predicted_label == real_label:
                if real_label not in found:
                    found[real_label] = 1
                else:
                    found[real_label] += 1
            if predicted_label != real_label:
                if real_label not in missed:
                    missed[real_label] = 1
                else:
                    missed[real_label] += 1
                if predicted_label not in misclassified:
                    misclassified[predicted_label] = 1
                else:
                    misclassified[predicted_label] += 1
        print "LSTM results"
        print "============================================================="
        print "Predicted correctly ", hits, "over ", guesses, " instances."
        recall_list = []
        recall_dict = {}
        precision_dict = {}
        found_labels = set(found.keys())
        missed_labels = set(missed.keys())
        all_labels = found_labels.union(missed_labels)
        for label in all_labels:
            no_of_finds = float((0 if label not in found else found[label]))
            no_of_missed = float((0 if label not in missed else missed[label]))
            no_of_misclassified = float((0 if label not in misclassified else misclassified[label]))
            recall = no_of_finds / (no_of_finds + no_of_missed)
            precision = no_of_finds / (no_of_finds + no_of_misclassified)
            recall_dict[label] = recall
            precision_dict[label] = precision
            recall_list.append(recall)
        print "Average recall ", np.mean(recall_list)
        if len(all_labels) == 2:  # compute F-1 score for binary classification
            for label in all_labels:
                # print "Precision for label ", label, " is : %f"%(precision_dict[label])
                # print "Recall for label ", label, " is : %f"%(precision_dict[label])
                print "F-1 score for label ", label, " is : ", 2 * (precision_dict[label] * recall_dict[label]) / (
                precision_dict[label] + recall_dict[label])
                # print "_____________________________________________________________"

            return precision_dict, recall_dict, recall_list, all_labels



def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def getDerivative(output, target, objective):
    if objective == "softmax_classification":
        return output - target

def get_balanced_training_set(training_set, no_of_classes):
    """
    Generator that returns items from training set
    equally balanced among classes
    """
    # make bucket of classes to sample from
    buckets = {}
    buckets_current_indexes ={}
    for i in range(0, no_of_classes):
        buckets[i] = []
        buckets_current_indexes[i] = 0
    for item in training_set:
        category = item.get_label()
        buckets[category].append(item)
    while True:
        for i in range(0,no_of_classes):
            if len(buckets[i]) == 0: # if a class has no representatives, continue
                continue
            if buckets_current_indexes[i] == len(buckets[i]):
                buckets_current_indexes[i] = 0
            yield buckets[i][buckets_current_indexes[i]]
            buckets_current_indexes[i] += 1

class InstanceNode:
    def __init__(self, label = None, sequence_control=None):
        """
        Create the instance node consisting of a label if any,
        and initiating cache and gradient values.

        The sequence_list consists of objects of class SequenceItem

        :param label:
        :type label:
        """
        self.label = label # an integer that represents the category of the item
        self.cache = {}
        self.gradient = {}
        self.sequence_list = []
        self.sequence_control = sequence_control # Stores the specific order by which the items were fed into the LSTM to update weights correctly
        self.helper_value = {}
        self.children_sequence = []

    def get_sequence_size(self):
        return len(self.sequence_list)

    def get_number_of_children(self):
        return self.get_sequence_size()

    def get_label(self):
        return self.label

    def get_sequence(self, sequence_function = None, help_value=None):
        """
        Return a specific sequence depending on children_sequence value and help_value
        :param help_value:
        :type help_value:
        :return:
        :rtype:
        """
        if sequence_function == "time":
            if help_value is not None and help_value.has_key('time'):
                return sorted([i for i in self.sequence_list if i.timestamp<help_value['time']], key=lambda x:x.timestamp)
            else:
                return self.sequence_list

        if sequence_function == "shuffle":
            random.shuffle(self.sequence_list)
            return self.sequence_list

        if sequence_function == "none":
            return sorted(self.sequence_list, key=lambda x:x.timestamp)

    def get_help_values(self):
        if len(self.helper_value):
            return self.helper_value
        else:
            return None

class SequenceItem:
    """
    Contains the item within a sequence which has a feature vector
    """

    def __init__(self, item_id, link_node_id, feature_vector = None, timestamp = None, action_time=None):
        """
        Initialize each sequence item with its owner details and
        associate a feature vector with it.

        :param item_id:
        :type item_id:
        :param link_node_id:
        :type link_node_id:
        :param feature_vector:
        :type feature_vector:
        """
        self.item_id = item_id
        self.link_node_id = link_node_id
        self.feature_vector = feature_vector
        self.action_time = action_time
        self.timestamp = timestamp

    def get_link_node(self):
        return self.link_node_id, self.action_time
        # return instance_dict.get(self.link_node_id)

    def get_feature_vector(self):
        return self.feature_vector