import os
import time
import json
import math
import torch
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

nlp = spacy.load("en_core_web_md")
os.environ['SENTENCE_TRANSFORMERS_HOME'] = "/path/to/working/dir/.cache/transformers"
sentence_embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def get_fact_candidates_similarity(fact, candidates):
    fact_embeddings = sentence_embedding_model.encode([fact])[0]
    candidates_embeddings = sentence_embedding_model.encode(candidates)
    sim_list = np.zeros(len(candidates))
    for index in range(len(candidates)):
        sim_list[index] = cos_sim(fact_embeddings, candidates_embeddings[index])
    return sim_list


def select_candidates(fact, candidates, topk=10):
    sim_list = get_fact_candidates_similarity(fact, candidates)
    sorted_index = np.argsort(sim_list)
    sorted_index = sorted_index[::-1][:topk]
    return [(candidates[x], sim_list[x]) for x in sorted_index]


def get_selected_candidates(test_path, output_path, option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        page_predictions = test_case[option]
        selected_candidates = []
        for sent_pages in page_predictions:
            selected_sent_candidates = []
            for sent, paragraphs in sent_pages:
                selected_sent_candidates.append((sent, select_candidates(sent, paragraphs)))
            selected_candidates.append(selected_sent_candidates)
        del test_case[option]
        test_case['selected_candidates'] = selected_candidates
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=4)

hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
entailment_tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name, cache_dir="/path/to/working/dir/.cache/transformers")
entailment_model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name,
                                                                      cache_dir="/path/to/working/dir/.cache/transformers")


def run_textual_entailment(batch_premise_hypothesis):
    # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contradiction"
    # },
    max_length = 512
    tokenized_input_seq_pair = entailment_tokenizer.batch_encode_plus(batch_premise_hypothesis,
                                                                      max_length=max_length,
                                                                      return_token_type_ids=True,
                                                                      truncation=True,
                                                                      return_tensors='pt',
                                                                      padding=True)
    input_ids = tokenized_input_seq_pair['input_ids']
    token_type_ids = tokenized_input_seq_pair['token_type_ids']
    attention_mask = tokenized_input_seq_pair['attention_mask']

    entailment_model.to('cuda')
    outputs = entailment_model(input_ids.to('cuda'),
                               attention_mask=attention_mask.to('cuda'),
                               token_type_ids=token_type_ids.to('cuda'),
                               labels=None)
    predicted_probability = torch.softmax(outputs[0], dim=1).cpu().tolist()  # batch_size only one
    return predicted_probability


def get_entailment_scores(test_path, output_path):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        selected_candidates = test_case['selected_candidates']
        predictions_scores = []
        in_batch_index = 0
        batch_premise_hypothesis = []
        for sent_candidates in selected_candidates:
            sent_scores = []
            for sent, candidates in sent_candidates:
                entailment_scores = []
                for candidate, similarity in candidates:
                    batch_premise_hypothesis.append((candidate, sent))
                    entailment_scores.append((candidate, similarity, in_batch_index))
                    in_batch_index += 1
                sent_scores.append((sent, entailment_scores))
            predictions_scores.append(sent_scores)
        predicted_probability = []
        small_batch_size = 2
        # the small batch size should be adjusted to GPU memory size
        batch_num = int(math.ceil(len(batch_premise_hypothesis) / small_batch_size))
        for batch_idx in range(batch_num):
            predicted_probability.extend(run_textual_entailment(batch_premise_hypothesis[batch_idx * small_batch_size:
                                                                                         batch_idx * small_batch_size +
                                                                                         small_batch_size]))
        for sent_scores in predictions_scores:
            for sent, entailment_scores in sent_scores:
                for index in range(len(entailment_scores)):
                    candidate, similarity, in_batch_index = entailment_scores[index]
                    entailment_scores[index] = (candidate, similarity, predicted_probability[in_batch_index])
        del test_case['selected_candidates']
        test_case['entailment_scores'] = predictions_scores
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=4)


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/Commonsense/'
    self_consistency_pages_path = dir_path + 'GPT-3/strategyqa_dev_self_consistency_gpt3_paragraphs.json'
    self_consistency_selected_path = dir_path + 'GPT-3/strategyqa_dev_self_consistency_gpt3_selected_candidates.json'
    self_consistency_entailment_path = dir_path + 'GPT-3/strategyqa_dev_self_consistency_gpt3_entailment.json'
    time_start = time.time()
    get_selected_candidates(self_consistency_pages_path, self_consistency_selected_path, option='sparse_retrieval')
    get_entailment_scores(self_consistency_selected_path, self_consistency_entailment_path)
    time_end = time.time()
    print('time:', time_end - time_start)

