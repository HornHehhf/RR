import json
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar


def get_answer_option(answer):
    answer = answer.lower()
    if "the answer is yes" in answer or "the answer is probably yes" in answer \
            or "the answer is most likely yes" in answer or answer.startswith('yes,') \
            or answer.startswith('yes.') or answer == 'yes':
        answer_option = "yes"
    elif "the answer is no" in answer or "the answer is probably no" in answer \
            or "the answer is most likely no" in answer \
            or answer.startswith('no,') or answer.startswith('no.') or answer == 'no':
        answer_option = "no"
    else:
        answer_option = "unclear"
    return answer_option


def get_majority_prediction(predictions):
    prediction_num = {'yes': 0, 'no': 0, 'unclear': 0}
    for prediction in predictions[:9]:
        prediction = prediction.replace('\n', '').strip()
        prediction = get_answer_option(prediction)
        prediction_num[prediction] += 1
    if prediction_num['yes'] > prediction_num['no']:
        return "yes"
    elif prediction_num['yes'] < prediction_num['no']:
        return "no"
    else:
        return "unclear"


def evaluate_qa(test_path, option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    total_case = 0.0
    correct_case = 0.0
    unclear_case = 0.0
    answer_map = {True: 'yes', False: 'no'}
    for idx, test_case in enumerate(test_data):
        prediction = test_case[option]
        ground_truth = test_case['answer']
        if isinstance(prediction, list):
            answer_option = get_majority_prediction(prediction)
        else:
            prediction = prediction.replace('\n', '').strip()
            answer_option = get_answer_option(prediction)
        if answer_option == answer_map[ground_truth]:
            correct_case += 1
        elif answer_option == 'unclear':
            unclear_case += 1
        total_case += 1
    print("Correct cases: ", correct_case)
    print("Unclear cases: ", unclear_case)
    print("Total cases: ", total_case)
    print('Accuracy: ', correct_case / total_case)
    print('Adjusted Accuracy', (correct_case + unclear_case / 2) / total_case)


def get_instance_level_performance(test_path, option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    acc_list = []
    answer_map = {True: 'yes', False: 'no'}
    for idx, test_case in enumerate(test_data):
        prediction = test_case[option]
        ground_truth = test_case['answer']
        if isinstance(prediction, list):
            answer_option = get_majority_prediction(prediction)
        else:
            prediction = prediction.replace('\n', '').strip()
            answer_option = get_answer_option(prediction)
        if answer_option == answer_map[ground_truth]:
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
    dir_path = '/path/to/working/dir/Commonsense/'
    few_shot_gpt3_path = dir_path + 'GPT-3/strategyqa_dev_few_shot_gpt3.json'
    chain_of_thought_gpt3_path = dir_path + 'GPT-3/strategyqa_dev_chain_of_thought_gpt3.json'
    self_consistency_gpt3_path = dir_path + 'GPT-3/strategyqa_dev_self_consistency_gpt3.json'
    weighting_baseline_path = dir_path + 'GPT-3/strategyqa_dev_gpt3_weighting_baseline.json'
    print('self consistency')
    evaluate_qa(self_consistency_gpt3_path, option='self_consistency_gpt3')
    print('weighting baseline')
    evaluate_qa(weighting_baseline_path, option='weighting_baseline')
    significant_testing(self_consistency_gpt3_path, weighting_baseline_path,
                        option_one='self_consistency_gpt3', option_two='weighting_baseline')
    significant_testing(few_shot_gpt3_path, chain_of_thought_gpt3_path,
                        option_one='few_shot_gpt3', option_two='chain_of_thought_gpt3')
