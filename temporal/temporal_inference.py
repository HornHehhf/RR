import json
import time
import numpy as np

from temporal_utils import get_answer, evaluate_qa


def check_conclusion_sentence(sent):
    sent = sent.replace('\n', '').strip().lower()
    if "answer is" in sent or sent.startswith('thus') or sent.startswith('therefore') or sent.startswith('so '):
        return True
    return False


def get_answer_score_weighting(sent_scores):
    topk = 1
    score_list = []
    entailment_scores_list = []
    contradict_scores_list = []
    for sent, entailment_scores in sent_scores:
        if check_conclusion_sentence(sent):
            continue
        max_score = -1.0
        max_entailment_score = 0.0
        max_contradiction_score = 0.0
        for kb_index in range(len(entailment_scores))[:topk]:
            if entailment_scores[kb_index][1] > max_score:
                max_score = entailment_scores[kb_index][1]
            if entailment_scores[kb_index][2][0] > max_entailment_score:
                max_entailment_score = entailment_scores[kb_index][2][0]
            if entailment_scores[kb_index][2][2] > max_contradiction_score:
                max_contradiction_score = entailment_scores[kb_index][2][2]
        score_list.append(max_score)
        entailment_scores_list.append(max_entailment_score)
        contradict_scores_list.append(max_contradiction_score)
    score_list = np.array(score_list)
    entailment_scores_list = np.array(entailment_scores_list)
    similarity_score = np.sum(score_list)
    entail_score = np.sum(entailment_scores_list)
    answer_score = similarity_score + entail_score
    return answer_score


def rerank_with_evidence(test_path, output_path):
    print('rerank with evidence')
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        predictions = test_case['self_consistency_gpt3']
        entailment_scores = test_case['entailment_scores']
        answer_score_list = []
        pred_answers = {}
        pred_answers_index = {}
        for gpt_index in range(len(predictions)):
            sent_scores = entailment_scores[gpt_index]
            answer_option = get_answer(predictions[gpt_index].replace('\n', '').strip())
            answer_score = get_answer_score_weighting(sent_scores)
            answer_score_list.append(answer_score)
            if answer_option not in pred_answers:
                pred_answers[answer_option] = answer_score
                pred_answers_index[answer_option] = [gpt_index]
            else:
                pred_answers[answer_option] += answer_score
                pred_answers_index[answer_option].append(gpt_index)
        max_answer_score = -np.inf
        max_answer_option = ""
        for answer_option in pred_answers.keys():
            if pred_answers[answer_option] > max_answer_score:
                max_answer_score = pred_answers[answer_option]
                max_answer_option = answer_option
        answer_score_list = np.array(answer_score_list)
        max_answer_index_list = pred_answers_index[max_answer_option]
        max_answer_index = max_answer_index_list[np.argmax(answer_score_list[max_answer_index_list])]
        test_case['weighting_baseline'] = predictions[max_answer_index]
        del test_case['entailment_scores']
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=4)


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/Temporal/'
    self_consistency_entailment_path = dir_path + 'GPT-3/implicit_temporal_self_consistency_gpt3_entailment.json'
    weighting_baseline_path = dir_path + 'GPT-3/implicit_temporal_gpt3_weighting_baseline.json'
    time_start = time.time()
    rerank_with_evidence(self_consistency_entailment_path, weighting_baseline_path)
    evaluate_qa(weighting_baseline_path, option='weighting_baseline')
    time_end = time.time()
    print('time:', time_end - time_start)


