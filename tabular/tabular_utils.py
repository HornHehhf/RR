import random
import json
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_answer_option(answer):
    answer = answer.lower()
    if "the answer is true" in answer or "the answer is probably true" in answer \
            or "the answer is most likely true" in answer or answer.startswith('true,') \
            or answer.startswith('true.') or answer == 'true':
        answer_option = "E"
    elif "the answer is false" in answer or "the answer is probably false" in answer \
            or "the answer is most likely false" in answer \
            or answer.startswith('false,') or answer.startswith('false.') or answer == 'false':
        answer_option = "C"
    else:
        answer_option = "unclear"
    return answer_option


def get_majority_prediction(predictions):
    prediction_num = {}
    for prediction in predictions:
        prediction = prediction.replace('\n', '').strip()
        prediction = get_answer_option(prediction)
        if prediction in prediction_num:
            prediction_num[prediction] += 1
        else:
            prediction_num[prediction] = 1
    prediction_num = sorted(prediction_num.items(), key=lambda item: item[1], reverse=True)
    # print(prediction_num)
    # print(prediction_num[0][0])
    return prediction_num[0][0]


def evaluate_qa(test_path, option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    total_case = 0.0
    correct_case = 0.0
    for idx, test_case in enumerate(test_data):
        prediction = test_case[option]
        ground_truth = test_case['label']
        if option in ['self_consistency_gpt3']:
            answer_option = get_majority_prediction(prediction)
        else:
            prediction = prediction.replace('\n', '').strip()
            answer_option = get_answer_option(prediction)
        if answer_option == ground_truth:
            correct_case += 1
        total_case += 1
    print("Correct cases: ", correct_case)
    print("Total cases: ", total_case)
    print('Accuracy: ', correct_case / total_case)


def get_instance_level_performance(test_path, option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    acc_list = []
    for idx, test_case in enumerate(test_data):
        prediction = test_case[option]
        ground_truth = test_case['label']
        if option in ['self_consistency_gpt3']:
            answer_option = get_majority_prediction(prediction)
        else:
            prediction = prediction.replace('\n', '').strip()
            answer_option = get_answer_option(prediction)
        if answer_option == ground_truth:
            acc_list.append(1)
        else:
            acc_list.append(0)
    print('instance num:', len(acc_list))
    print('accuracy:', np.mean(acc_list))
    return acc_list


def get_contingency_tabel(exact_match_list_one, exact_match_list_two):
    table = np.zeros((2, 2))
    for index in range(len(exact_match_list_one)):
        item_one = exact_match_list_one[index]
        item_two = exact_match_list_two[index]
        table[item_one, item_two] += 1
    print(table)
    return table


def significant_testing(file_one, file_two, option_one, option_two):
    print('get {0} performance'.format(option_one))
    acc_list_one = get_instance_level_performance(file_one, option_one)
    print('get {0} performance'.format(option_two))
    acc_list_two = get_instance_level_performance(file_two, option_two)
    print('{0} vs {1} (McNemar test acc)'.format(option_one, option_two))
    print(mcnemar(get_contingency_tabel(acc_list_one, acc_list_two)))
    _, pval_acc = stats.ttest_rel(acc_list_one, acc_list_two)
    print('{0} vs {1} (acc): {2}'.format(option_one, option_two, pval_acc))
    return pval_acc


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/Tabular/'
    zero_shot_gpt3_path = dir_path + 'GPT-3/binary_dev_zero_shot_gpt3.json'
    few_shot_gpt3_path = dir_path + 'GPT-3/binary_dev_few_shot_gpt3.json'
    chain_of_thought_gpt3_path = dir_path + 'GPT-3/binary_dev_chain_of_thought_gpt3.json'
    self_consistency_gpt3_path = dir_path + 'GPT-3/binary_dev_self_consistency_gpt3.json'
    weighting_baseline_path = dir_path + 'GPT-3/binary_dev_gpt3_weighting_baseline.json'
    print('self consistency')
    evaluate_qa(self_consistency_gpt3_path, 'self_consistency_gpt3')
    significant_testing(self_consistency_gpt3_path, weighting_baseline_path,
                        option_one='self_consistency_gpt3', option_two='weighting_baseline')
    significant_testing(few_shot_gpt3_path, chain_of_thought_gpt3_path,
                        option_one='few_shot_gpt3', option_two='chain_of_thought_gpt3')
