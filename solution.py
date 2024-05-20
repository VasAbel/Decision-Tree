import numpy as np #(működik a Moodle-ben is)
import math

######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    entropy = 0
    if n_cat1 == 0 or n_cat2 == 0:
        return 0.0
    sum_of_records = n_cat1 + n_cat2
    cat1_poss = n_cat1/sum_of_records
    cat2_poss = n_cat2/sum_of_records
    entropy = -(cat1_poss * math.log2(cat1_poss) + cat2_poss * math.log2(cat2_poss))
    return entropy

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list,
                        labels: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    features = np.array(features)
    labels = np.array(labels)
    cat1_l = 0
    cat2_l = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            cat1_l += 1
        else:
            cat2_l += 1
    l = cat1_l + cat2_l
    entropy_l = get_entropy(cat1_l, cat2_l)
    information = float('-inf')
    ###feature indexnél vagyunk
    for i in range(features.shape[1]):
        feature_values = [row[i] for row in features]
        sorted_values = sorted(feature_values)
        ###adott érték mentén vagyunk
        for j in range(len(sorted_values) - 1):
            split_value = (sorted_values[j] + sorted_values[j + 1]) / 2
            n_lower_cat1 = 0
            n_lower_cat2 = 0
            n_higher_cat1 = 0
            n_higher_cat2 = 0
            ###adott rekord értékét vizsgáljuk
            for k in range(len(feature_values)):
                if feature_values[k] <= split_value:
                    if labels[k] == 0:
                        n_lower_cat1 += 1
                    else:
                        n_lower_cat2 += 1
                else:
                    if labels[k] == 0:
                        n_higher_cat1 += 1
                    else:
                        n_higher_cat2 += 1
            curr_information = entropy_l - (((n_lower_cat1 + n_lower_cat2) / l) * get_entropy(n_lower_cat1, n_lower_cat2) + ((n_higher_cat1 + n_higher_cat2) / l) * get_entropy(n_higher_cat1, n_higher_cat2))
            if curr_information > information:
                information = curr_information
                best_separation_feature = i
                best_separation_value = split_value

    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################
def main():
    train_file = np.loadtxt('train.csv', delimiter=',')
    features = train_file[:, :-1]
    labels = train_file[:, -1].astype(int)
    d_tree = decision_tree(features, labels)
    test_file = np.loadtxt('test.csv', delimiter=',')

    results_file = open('results.csv', 'w')

    for row in test_file:
        output = decide_output(row, d_tree)
        results_file.write(str(output) + '\n')

    results_file.close()

    return 0

def decision_tree(features, labels):
    cat1 = sum(labels == 0)
    cat2 = sum(labels == 1)
    if get_entropy(cat1, cat2) == 0.0:
        return labels[0]

    split_feature, split_value = get_best_separation(features, labels)

    node = (split_feature, split_value)
    first_side = features[:, split_feature] <= split_value
    second_side = features[:, split_feature] > split_value

    left_tree = decision_tree(features[first_side], labels[first_side])
    right_tree = decision_tree(features[second_side], labels[second_side])

    return {node: (left_tree, right_tree)}

def decide_output(row, tree):
    if type(tree) is not dict:
        return tree
    current_node = list(tree.keys())[0]
    feature = current_node[0]
    value = current_node[1]
    left_child = tree[current_node][0]
    right_child = tree[current_node][1]
    if row[feature] <= value:
        return decide_output(row, left_child)
    else:
        return decide_output(row, right_child)


if __name__ == "__main__":
    main()
