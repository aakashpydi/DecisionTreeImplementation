import sys
import math
import pandas as pd

#import matplotlib.pyplot as plt

# from sklearn import tree
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelEncoder
# from StringIO import StringIO

attribute_headers = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']

class AttributeValueHist:
    def __init__(self, name, count):
        self.attribute_name = name
        self.count = count
        self.income_less_50_count = 0.0
        self.income_greater_50_count = 0.0

    def __str__(self):
        return "Name: "+ self.attribute_name + "\tCount: "+ str(self.count) + "\t<=50K: "+str(self.income_less_50_count) + "\t>50K: " + str(self.income_greater_50_count)

    def incrementCount(self, label):
        self.count += 1
        if label == "<=50K":
            self.income_less_50_count += 1.0
        else:
            self.income_greater_50_count += 1.0

class DecisionTreeNode:
    node_count = 0
    depthLimit = 0

    def __init__(self, data, depth):
        self.isLeaf = False
        self.leftDecisionTreeNode = None
        self.rightDecisionTreeNode = None
        self.attribute_value_used_for_split = None
        self.class_label_hist = [0.0, 0.0]
        self.node_prune_accuracy_score = -1
        self.depth = depth
        self.data = data

        if model == "depth" and depth > DecisionTreeNode.depthLimit:
            print str(depth) + " " + str(DecisionTreeNode.depthLimit)
            self.isLeaf = False
        else:
            self.isLeaf = True
            DecisionTreeNode.node_count += 1
            value_set = []
            for tuple in data:
                for counter, element in enumerate(tuple):
                    if counter == 8:
                        if element == "<=50K":
                            self.class_label_hist[0] += 1.0
                        else:
                            self.class_label_hist[1] += 1.0
                    else:
                        value_set.append(attribute_headers[counter] + "=" + str(element))
            value_set = set(value_set)

            self.value_set_l = list(value_set)
            self.attributeValuesList = []
            for s in self.value_set_l:
                self.attributeValuesList.append(AttributeValueHist(s, 0))
            self.initializeAttributeHistograms()
            self.computeInfomationGains()
            if model == "depth" and depth == DecisionTreeNode.depthLimit:
                self.isLeaf = True
            if self.isLeaf == False:
                self.initializeSubTrees()

    def initializeAttributeHistograms(self):
        for tuple in self.data:
            for counter, element in enumerate(tuple):
                if counter == 8:
                    continue
                index = self.value_set_l.index(attribute_headers[counter] + "=" + str(element))
                self.attributeValuesList[index].incrementCount(tuple[8])

    def computeInfomationGains(self):
        ## first compute entropy
        total_values = self.class_label_hist[0] + self.class_label_hist[1]
        entropy_value = 0.0

        p_1 = self.class_label_hist[0]/total_values
        p_2 = self.class_label_hist[1]/total_values

        if p_1 != 0.0:
            entropy_value +=  (-1 * p_1) * math.log(p_1, 2)
        if p_2 != 0.0:
            entropy_value +=  (-1 * p_2) * math.log(p_2, 2)

        if entropy_value == 0:
            self.isLeaf = True
            return

        max_information_gain = -10.0
        max_info_index = -1

        for i, attributeHist in enumerate(self.attributeValuesList):
            # calculate for potential left decision subtree with value PRESENT
            #print attributeHist.attribute_name

            p_1 = attributeHist.income_less_50_count/attributeHist.count
            p_2 = attributeHist.income_greater_50_count/attributeHist.count

            if p_1 == 0:
                left_income_less_50_entropy = 0
            else:
                left_income_less_50_entropy = (-1 * p_1) * math.log(p_1, 2)

            if p_2 == 0:
                left_income_greater_50_entropy = 0
            else:
                left_income_greater_50_entropy = (-1 * p_2) * math.log(p_2, 2)

            left_entropy = left_income_less_50_entropy + left_income_greater_50_entropy

            #calculate entropy for potential right decision subtree with value ABSENT
            right_total_values = total_values - attributeHist.count
            if right_total_values == 0:
                p_1 = 0
                p_2 = 0
            else:
                p_1 = (self.class_label_hist[0] - attributeHist.income_less_50_count)/right_total_values
                p_2 = (self.class_label_hist[1] - attributeHist.income_greater_50_count)/right_total_values

            if p_1 == 0:
                right_income_less_50_entropy = 0
            else:
                right_income_less_50_entropy = (-1 * p_1) * math.log(p_1, 2)

            if p_2 == 0:
                right_income_greater_50_entropy = 0
            else:
                right_income_greater_50_entropy = (-1 * p_2) * math.log(p_2, 2)

            right_entropy = right_income_less_50_entropy + right_income_greater_50_entropy

            #weighted average
            child_entropy_value = (attributeHist.count/total_values)*left_entropy + (right_total_values/total_values)*right_entropy
            if entropy_value - child_entropy_value > max_information_gain:
                max_information_gain = entropy_value - child_entropy_value
                max_info_index = i

            # print "child entropy value:  " + str(child_entropy_value)
            # print "Information Gain: " + str(entropy_value - child_entropy_value) +"\n"
        # print "FOUND:\t\t" + str(max_information_gain) + "  " + str(self.attributeValuesList[max_info_index]) +"\n"
        if max_information_gain > 0.0:
            self.isLeaf = False
            self.attribute_value_used_for_split = self.attributeValuesList[max_info_index]

        else:
            self.isLeaf = True

    def initializeSubTrees(self):
        leftTreeData = []
        rightTreeData= []

        for tuple in self.data:
            isTupleInserted = False

            for counter, element in enumerate(tuple):
                if counter == 8:
                    continue
                att_val = attribute_headers[counter] + "=" + str(element)
                if att_val == self.attribute_value_used_for_split.attribute_name:
                    leftTreeData.append(tuple)
                    isTupleInserted = True
            if isTupleInserted == False:
                rightTreeData.append(tuple)

        self.leftDecisionTreeNode = DecisionTreeNode(leftTreeData, self.depth + 1)
        self.rightDecisionTreeNode = DecisionTreeNode(rightTreeData, self.depth + 1)


    def predictTuple(self, dataTuple):
        if self.isLeaf == True:
            #print "FOUND LEAF"
            # print self.class_label_hist
            # print self.leftDecisionTreeNode
            # print self.rightDecisionTreeNode
            #print "PREDICTING USING: "+ str(self.class_label_hist)
            if self.class_label_hist[0] > self.class_label_hist[1] :
                return "<=50K"
            else:
                return ">50K"
        else:
            #print "ENTERING HERE"
            #print "tesssssssssst" + str(self.class_label_hist) +" : isLeaf" + str(self.isLeaf)
            found = False
            for i, element in enumerate(dataTuple):
                current = attribute_headers[i]+"="+element
                if current == self.attribute_value_used_for_split.attribute_name:
                    return self.leftDecisionTreeNode.predictTuple(dataTuple)
            if found == False:
                return self.rightDecisionTreeNode.predictTuple(dataTuple)

    def prune(self):
        if self.isLeaf == True:
            return self.node_prune_accuracy_score
        else:
            child_accuracy = self.leftDecisionTreeNode.prune() + self.rightDecisionTreeNode.prune()
            if child_accuracy < self.node_prune_accuracy_score:
                self.isLeaf = True
                self.leftDecisionTreeNode = None
                self.rightDecisionTreeNode = None
                DecisionTreeNode.node_count -= 2
            return self.node_prune_accuracy_score
            # else:
            #     return child_accuracy

    def pruningInitialize(self, X_validation):
        #print "Debug: "+str(self.class_label_hist)
        if self.class_label_hist[0] > self.class_label_hist[1] :
            prediction = "<=50K"
        else:
            prediction = ">50K"

        node_accuracy_score = 0.0
        for tuple in X_validation:
            if(tuple[8] == prediction):
                node_accuracy_score += 1
        self.node_prune_accuracy_score = node_accuracy_score

        if self.isLeaf == False:
            leftTreeData = []
            rightTreeData= []

            for tuple in X_validation:
                isTupleInserted = False

                for counter, element in enumerate(tuple):
                    if counter == 8:
                        continue
                    att_val = attribute_headers[counter] + "=" + str(element)
                    if att_val == self.attribute_value_used_for_split.attribute_name:
                        leftTreeData.append(tuple)
                        isTupleInserted = True
                if isTupleInserted == False:
                    rightTreeData.append(tuple)
            #print "YOOOO: " + str(len(leftTreeData)) + " : " + str(len(rightTreeData))
            self.leftDecisionTreeNode.pruningInitialize(leftTreeData)
            self.rightDecisionTreeNode.pruningInitialize(rightTreeData)

    # def prune(self):
    #     if self.isLeaf == False:
    #         self.leftDecisionTreeNode.prune()
    #         self.rightDecisionTreeNode.prune()
    #         child_accuracy = self.leftDecisionTreeNode.node_prune_accuracy_score + self.rightDecisionTreeNode.node_prune_accuracy_score
    #         if self.node_prune_accuracy_score > child_accuracy:
    #             self.isLeaf = True
    #             DecisionTreeNode.node_count -= 2
    #             self.leftDecisionTreeNode = None
    #             self.rightDecisionTreeNode = None

    def printTreeInfo(self):
        if self != None:
            #if self.isLeaf == True:
                #print "Found LEAF: " + str(self.class_label_hist)
            #print self.node_prune_accuracy_score
            if self.leftDecisionTreeNode != None:
                self.leftDecisionTreeNode.printTreeInfo()
            if self.rightDecisionTreeNode != None:
                self.rightDecisionTreeNode.printTreeInfo()

def printAccuracy(X, dtNode, stringToPrint):
    correct_predictions = 0.0
    for tuple in X:
        prediction = dtNode.predictTuple(tuple)
        if tuple[8] == prediction:
            correct_predictions += 1.0
    print stringToPrint + str(correct_predictions/float(len(X)))

def getAccuracy(X, dtNode):
    correct_predictions = 0.0
    for tuple in X:
        prediction = dtNode.predictTuple(tuple)
        if tuple[8] == prediction:
            correct_predictions += 1.0
    return correct_predictions/float(len(X))


#### Main Script Begins
train_filename = sys.argv[1]
test_filename = sys.argv[2]
model = sys.argv[3]

data = pd.read_csv(train_filename, sep=', ', quotechar='"', header=None, engine='python')
X = data.as_matrix()

X_train = None
X_test = None
X_validation = None
DecisionTreeNode.depthLimit = len(X)     #default is max possible depth

data_test = pd.read_csv(test_filename, sep=', ', quotechar='"', header=None, engine='python')
X_test = data_test.as_matrix()

if(model == "vanilla"):
    # depth limit is default max
    train_percentage_to_use = float(sys.argv[4])
    train_percentage_to_use /= 100.0
    X_train = X[:int(train_percentage_to_use*len(X))]
elif(model == "depth"):
    train_percentage_to_use = float(sys.argv[4])
    train_percentage_to_use /= 100.0
    train_to_row_index = int(train_percentage_to_use*len(X))
    X_train = X[:train_to_row_index]

    validation_percentage_to_use = float(sys.argv[5])
    validation_percentage_to_use /= 100.0
    validation_to_index = train_to_row_index + int((len(X) * validation_percentage_to_use))
    X_validation = X[train_to_row_index: validation_to_index]
    DecisionTreeNode.depthLimit = int(sys.argv[6])
elif(model == "prune"):
    #depthLimit is default max
    train_percentage_to_use = float(sys.argv[4])
    train_percentage_to_use /= 100.0
    train_to_row_index = int(train_percentage_to_use*len(X))
    X_train = X[:train_to_row_index]

    validation_percentage_to_use = float(sys.argv[5])
    validation_percentage_to_use /= 100.0
    validation_to_index = train_to_row_index + int((len(X) * validation_percentage_to_use))
    X_validation = X[train_to_row_index: validation_to_index]


dt = DecisionTreeNode(X_train, 0)
#print "Node Count Before Prune: " + str(DecisionTreeNode.node_count)
if model == "vanilla":
    printAccuracy(X_train, dt, "Training set accuracy: ")
    printAccuracy(X_test, dt, "Test set accuracy: ")
elif model == "depth":
    printAccuracy(X_train, dt, "Training set accuracy: ")
    printAccuracy(X_validation, dt, "Validation set accuracy: ")
    printAccuracy(X_test, dt, "Test set accuracy: ")
elif model == "prune":
    #printAccuracy(X_train, dt, "Before Prune Training set accuracy: ")
    #printAccuracy(X_test, dt, "Before Prune Test set accuracy: ")
    dt.pruningInitialize(X_validation)
    dt.prune()
    #print "Node Count After Prune: " + str(DecisionTreeNode.node_count)
    printAccuracy(X_train, dt, "Training set accuracy: ")
    printAccuracy(X_test, dt, "Test set accuracy: ")

#################################################################
#################################################################
########## Following Functions Used for Analysis
# def vanilla_analysis(X_full_train_data_set, X_test):
#     number_of_nodes = []
#     training_set_accuracies = []
#     test_set_accuracies = []
#     training_set_percentages = [2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
#
#     for percentage_to_use in training_set_percentages:
#         percentage_to_use /= 100.0
#         train_to_row_index = int(percentage_to_use * len(X_full_train_data_set))
#         X_train = X_full_train_data_set[:train_to_row_index]
#
#         dt = DecisionTreeNode(X_train, 0)
#         #print DecisionTreeNode.node_count
#         number_of_nodes.append(DecisionTreeNode.node_count)
#         training_set_accuracies.append(getAccuracy(X_train, dt))
#         test_set_accuracies.append(getAccuracy(X_test, dt))
#         DecisionTreeNode.node_count = 0
#
#     plt.plot(training_set_percentages, training_set_accuracies)
#     plt.plot(training_set_percentages, test_set_accuracies)
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Accuracies')
#     plt.legend(['Training Set', 'Test Set'], loc='upper right')
#     plt.show()
#
#     plt.plot(training_set_percentages, number_of_nodes, 'ro')
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Number of Nodes')
#     plt.show()
# def depth_analysis(X_full_train_data_set, X_test):
#     number_of_nodes = []
#     training_set_accuracies = []
#     test_set_accuracies = []
#     optimal_depths = []
#     training_set_percentages = [2.0, 10.0, 20.0, 30.0, 40.0, 60.0]
#     possible_depth_values = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
#
#     for percentage_to_use in training_set_percentages:
#         print percentage_to_use
#         percentage_to_use /= 100.0
#         train_to_row_index = int(percentage_to_use * len(X_full_train_data_set))
#         X_train = X_full_train_data_set[:train_to_row_index]
#
#         validation_percentage_to_use = 0.4
#         validation_to_index = train_to_row_index + int((len(X) * validation_percentage_to_use))
#         X_validation = X[train_to_row_index: validation_to_index]
#
#         best_accuracy = 0.0
#         best_depth = 1
#         for depth in possible_depth_values:
#             DecisionTreeNode.depthLimit = depth
#             dt = DecisionTreeNode(X_train, 0)
#             accuracy = getAccuracy(X_validation, dt)
#             print "-----" +str(depth) +" : " + str(accuracy)
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_depth = depth
#         DecisionTreeNode.node_count = 0
#         optimal_depths.append(best_depth)
#         DecisionTreeNode.depthLimit = best_depth
#
#         dt = DecisionTreeNode(X_train, 0)
#         #print DecisionTreeNode.node_count
#         number_of_nodes.append(DecisionTreeNode.node_count)
#         training_set_accuracies.append(getAccuracy(X_train, dt))
#         test_set_accuracies.append(getAccuracy(X_test, dt))
#         DecisionTreeNode.node_count = 0
#
#     plt.plot(training_set_percentages, training_set_accuracies)
#     plt.plot(training_set_percentages, test_set_accuracies)
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Accuracies')
#     plt.legend(['Training Set', 'Test Set'], loc='upper right')
#     plt.show()
#
#     plt.plot(training_set_percentages, number_of_nodes, 'ro')
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Number of Nodes')
#     plt.show()
#
#     plt.plot(training_set_percentages, optimal_depths, 'ro')
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Optimal Depths')
#     plt.show()
#
# def prune_analysis(X_full_train_data_set, X_test):
#     number_of_nodes = []
#     training_set_accuracies = []
#     test_set_accuracies = []
#     training_set_percentages = [2.0, 10.0, 20.0, 30.0, 40.0, 60.0]
#
#     for percentage_to_use in training_set_percentages:
#         print percentage_to_use
#         percentage_to_use /= 100.0
#         train_to_row_index = int(percentage_to_use * len(X_full_train_data_set))
#         X_train = X_full_train_data_set[:train_to_row_index]
#
#         validation_percentage_to_use = 0.4
#         validation_to_index = train_to_row_index + int((len(X) * validation_percentage_to_use))
#         X_validation = X[train_to_row_index: validation_to_index]
#
#         dt = DecisionTreeNode(X_train, 0)
#         dt.pruningInitialize(X_validation)
#         dt.prune()
#
#         number_of_nodes.append(DecisionTreeNode.node_count)
#         training_set_accuracies.append(getAccuracy(X_train, dt))
#         test_set_accuracies.append(getAccuracy(X_test, dt))
#         DecisionTreeNode.node_count = 0
#
#     plt.plot(training_set_percentages, training_set_accuracies)
#     plt.plot(training_set_percentages, test_set_accuracies)
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Accuracies')
#     plt.legend(['Training Set', 'Test Set'], loc='upper right')
#     plt.show()
#
#     plt.plot(training_set_percentages, number_of_nodes, 'ro')
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Number of Nodes')
#     plt.show()
#
# def sklearnVanillaAnalysis(X_full_train_data_set, X_test):
#     number_of_nodes = []
#     training_set_accuracies = []
#     test_set_accuracies = []
#     training_set_percentages = [2.0, 10.0, 20.0, 30.0, 40.0, 60.0]
#
#     value_set = []
#     for tuple in X_full_train_data_set:
#         for counter, element in enumerate(tuple):
#             value_set.append(str(element))
#     for tuple in X_test:
#         for counter, element in enumerate(tuple):
#             value_set.append(str(element))
#     value_set = set(value_set)
#     value_set_l = list(value_set)
#     le = LabelEncoder()
#     le.fit(value_set_l)
#
#     for i, tuple in enumerate(X_full_train_data_set):
#         X_full_train_data_set[i] = map(int, le.transform(tuple))
#
#     for i, tuple in enumerate(X_test):
#         X_test[i] = map(int, le.transform(tuple))
#
#     for percentage_to_use in training_set_percentages:
#         percentage_to_use /= 100.0
#         train_to_row_index = int(percentage_to_use * len(X_full_train_data_set))
#         X_train = X_full_train_data_set[:train_to_row_index]
#
#         clf = tree.DecisionTreeClassifier(criterion="entropy")
#         clf = clf.fit(X_train[:, [0,1,2,3,4,5,6,7]], X_train[:, [8]].tolist())
#         #print DecisionTreeNode.node_count
#         number_of_nodes.append(clf.tree_.node_count)
#         training_set_accuracies.append(clf.score(X_train[:, [0,1,2,3,4,5,6,7]], X_train[:, [8]].tolist()))
#         test_set_accuracies.append(clf.score(X_test[:, [0,1,2,3,4,5,6,7]], X_test[:, [8]].tolist()))
#
#     plt.plot(training_set_percentages, training_set_accuracies)
#     plt.plot(training_set_percentages, test_set_accuracies)
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Accuracies')
#     plt.legend(['Training Set', 'Test Set'], loc='upper right')
#     plt.title('SKLEARN Accuracies Test')
#     plt.show()
#
#     plt.plot(training_set_percentages, number_of_nodes, 'ro')
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Number of Nodes')
#     plt.title('SKLEARN Number of Nodes Test')
#     plt.show()
#
# def sklearnDepthAnalysis(X_full_train_data_set, X_test):
#     number_of_nodes = []
#     training_set_accuracies = []
#     test_set_accuracies = []
#     optimal_depths = []
#     training_set_percentages = [2.0, 10.0, 20.0, 30.0, 40.0, 60.0]
#     possible_depth_values = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
#
#     value_set = []
#     for tuple in X_full_train_data_set:
#         for counter, element in enumerate(tuple):
#             value_set.append(str(element))
#     for tuple in X_test:
#         for counter, element in enumerate(tuple):
#             value_set.append(str(element))
#     value_set = set(value_set)
#     value_set_l = list(value_set)
#     le = LabelEncoder()
#     le.fit(value_set_l)
#
#     for i, tuple in enumerate(X_full_train_data_set):
#         X_full_train_data_set[i] = map(int, le.transform(tuple))
#
#     for i, tuple in enumerate(X_test):
#         X_test[i] = map(int, le.transform(tuple))
#
#     for percentage_to_use in training_set_percentages:
#         print percentage_to_use
#         percentage_to_use /= 100.0
#         train_to_row_index = int(percentage_to_use * len(X_full_train_data_set))
#         X_train = X_full_train_data_set[:train_to_row_index]
#
#         validation_percentage_to_use = 0.4
#         validation_to_index = train_to_row_index + int((len(X) * validation_percentage_to_use))
#         X_validation = X[train_to_row_index: validation_to_index]
#
#         best_accuracy = 0.0
#         best_depth = 1
#         for depth in possible_depth_values:
#             DecisionTreeNode.depthLimit = depth
#             clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth)
#             clf = clf.fit(X_train[:, [0,1,2,3,4,5,6,7]], X_train[:, [8]].tolist())
#             accuracy = clf.score(X_validation[:, [0,1,2,3,4,5,6,7]], X_validation[:, [8]].tolist())
#             print "----- SKLEARN ---" +str(depth) +" : " + str(accuracy)
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_depth = depth
#         optimal_depths.append(best_depth)
#
#         clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=best_depth)
#         clf = clf.fit(X_train[:, [0,1,2,3,4,5,6,7]], X_train[:, [8]].tolist())
#
#
#         number_of_nodes.append(clf.tree_.node_count)
#         training_set_accuracies.append(clf.score(X_train[:, [0,1,2,3,4,5,6,7]], X_train[:, [8]].tolist()))
#         test_set_accuracies.append(clf.score(X_test[:, [0,1,2,3,4,5,6,7]], X_test[:, [8]].tolist()))
#
#     plt.plot(training_set_percentages, training_set_accuracies)
#     plt.plot(training_set_percentages, test_set_accuracies)
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Accuracies')
#     plt.legend(['Training Set', 'Test Set'], loc='upper right')
#     plt.title('SKLEARN Accuracies Test')
#     plt.show()
#
#     plt.plot(training_set_percentages, number_of_nodes, 'ro')
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Number of Nodes')
#     plt.title('SKLEARN Number of Nodes Test')
#     plt.show()
#
#     plt.plot(training_set_percentages, optimal_depths, 'ro')
#     plt.xlabel('Training Set Percentages (as %)')
#     plt.ylabel('Optimal Depths')
#     plt.show()
#
# def runSklearnDT(X_train, X_test, X_validation, depth):
#     value_set = []
#     for tuple in X_train:
#         for counter, element in enumerate(tuple):
#             value_set.append(str(element))
#     for tuple in X_test:
#         for counter, element in enumerate(tuple):
#             value_set.append(str(element))
#     if X_validation is not None:
#         for tuple in X_validation:
#             for counter, element in enumerate(tuple):
#                 value_set.append(str(element))
#     value_set = set(value_set)
#     value_set_l = list(value_set)
#     le = LabelEncoder()
#     le.fit(value_set_l)
#     for i, tuple in enumerate(X_train):
#         X_train[i] = map(int, le.transform(tuple))
#
#     for i, tuple in enumerate(X_test):
#         X_test[i] = map(int, le.transform(tuple))
#
#     if X_validation is not None:
#         for i, tuple in enumerate(X_validation):
#             X_validation[i] = map(int, le.transform(tuple))
#
#     clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth)
#     clf = clf.fit(X_train[:, [0,1,2,3,4,5,6,7]], X_train[:, [8]].tolist())
#     print "SKLEARN Training Set Accuracy: " + str(clf.score(X_train[:, [0,1,2,3,4,5,6,7]], X_train[:, [8]].tolist()))
#     if X_validation is not None:
#         print "SKLEARN Validation Set Accuracy: " + str(clf.score(X_validation[:, [0,1,2,3,4,5,6,7]], X_validation[:, [8]].tolist()))
#     print "SKLEARN Test Set Accuracy: " + str(clf.score(X_test[:, [0,1,2,3,4,5,6,7]], X_test[:, [8]].tolist()))
# ######### End Analysis Functions
