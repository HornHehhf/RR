""" Adopted from Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import json
import spacy
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

nlp = spacy.load('en_core_web_md')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_answer(prediction):
    if 'answer is' not in prediction:
        nlp_text = nlp(prediction).sents
        sentences = [str(sent).strip() for sent in nlp_text]
        return sentences[-1]
    ans_index = prediction.index('answer is') + 9
    return prediction[ans_index:]


def get_majority_prediction(predictions):
    prediction_num = {}
    for prediction in predictions:
        prediction = prediction.replace('\n', '').strip()
        prediction = get_answer(prediction)
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
    f1 = exact_match = total = 0
    for idx, test_case in enumerate(test_data):
        if 'implicit' in test_path:
            if idx < 6:
                continue
        total += 1
        # print('\nquestion:', test_case['Question'])
        # print('answer:', test_case['Gold answer'])
        prediction = test_case[option]
        ground_truths = test_case['Gold answer']
        if option in ['zero_shot_gpt3', 'few_shot_gpt3']:
            prediction = prediction.replace('\n', '').strip()
        elif option in ['chain_of_thought_gpt3', 'weighting_baseline']:
            prediction = prediction.replace('\n', '').strip()
            prediction = get_answer(prediction)
        elif option in ['self_consistency_gpt3']:
            prediction = get_majority_prediction(prediction)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print('qa num:', total)
    print('exact match:', exact_match)
    print('f1:', f1)


def get_instance_level_performance(test_path, option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    exact_match_list = []
    f1_list = []
    for idx, test_case in enumerate(test_data):
        if idx < 6:
            continue
        prediction = test_case[option]
        ground_truths = test_case['Gold answer']
        if option in ['zero_shot_gpt3', 'few_shot_gpt3']:
            prediction = prediction.replace('\n', '').strip()
        elif option in ['chain_of_thought_gpt3', 'weighting_baseline']:
            prediction = prediction.replace('\n', '').strip()
            prediction = get_answer(prediction)
        elif option in ['self_consistency_gpt3']:
            prediction = get_majority_prediction(prediction)
        exact_match_list.append(metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths))
        f1_list.append(metric_max_over_ground_truths(
            f1_score, prediction, ground_truths))
    print('instance num:', len(exact_match_list), len(f1_list))
    print('exact match:', np.mean(exact_match_list))
    print('f1:', np.mean(f1_list))
    return exact_match_list, f1_list


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
    exact_match_list_one, f1_list_one = get_instance_level_performance(file_one, option_one)
    print('get {0} performance'.format(option_two))
    exact_match_list_two, f1_list_two = get_instance_level_performance(file_two, option_two)
    print('{0} vs {1} (McNemar test exact match)'.format(option_one, option_two))
    print(mcnemar(get_contingency_tabel(exact_match_list_one, exact_match_list_two)))
    _, pval_exact_match = stats.ttest_rel(exact_match_list_one, exact_match_list_two)
    print('{0} vs {1} (exact match): {2}'.format(option_one, option_two, pval_exact_match))
    _, pval_f1 = stats.ttest_rel(f1_list_one, f1_list_two)
    print('{0} vs {1} (f1): {2}'.format(option_one, option_two, pval_f1))
    return pval_exact_match, pval_f1


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/Temporal/'
    zero_shot_gpt3_path = dir_path + 'GPT-3/implicit_temporal_zero_shot_gpt3.json'
    few_shot_gpt3_path = dir_path + 'GPT-3/implicit_temporal_few_shot_gpt3.json'
    chain_of_thought_gpt3_path = dir_path + 'GPT-3/implicit_temporal_chain_of_thought_gpt3.json'
    self_consistency_gpt3_path = dir_path + 'GPT-3/implicit_temporal_self_consistency_gpt3.json'
    weighting_baseline_path = dir_path + 'GPT-3/implicit_temporal_gpt3_weighting_baseline.json'
    significant_testing(self_consistency_gpt3_path, weighting_baseline_path,
                        option_one='self_consistency_gpt3', option_two='weighting_baseline')
    significant_testing(few_shot_gpt3_path, chain_of_thought_gpt3_path,
                        option_one='few_shot_gpt3', option_two='chain_of_thought_gpt3')
