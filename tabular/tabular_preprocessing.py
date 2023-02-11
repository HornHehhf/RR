import pandas as pd
import json
import ast
import random


from tabular_utils import set_random_seed

dir_path = '/path/to/working/dir/Tabular/'


def process_training_data():
    template_path = dir_path + 'relation_templates.json'
    relation_data_path = dir_path + 'train_with_lstmrels.csv'
    data_path = dir_path + 'infotabs_train_relations_shuffled.json'
    template_file = open(template_path, 'r')
    relation_templates = json.load(template_file)
    relation_data = pd.read_csv(relation_data_path)
    relation_data_columns = ['table_id', 'premise', 'hypothesis', 'label', 'Rel_with_toks']
    id_2_label = {0: 'C', 1: 'N', 2: 'E'}
    data = []
    for index in range(len(relation_data)):
        cur_point = {}
        for key in relation_data_columns:
            if key == 'label':
                cur_point[key] = id_2_label[relation_data.loc[index, key]]
            elif key == 'Rel_with_toks':
                single_word_relations = set()
                if relation_data.loc[index, key] == relation_data.loc[index, key]:
                    token_relations = ast.literal_eval(relation_data.loc[index, key])
                    for token_relation in token_relations:
                        for rel_type in token_relation[3]:
                            relation = (token_relation[0][0], token_relation[1][0], relation_templates[rel_type])
                            relation = relation[0] + ' ' + relation[2] + ' ' + relation[1]
                            single_word_relations.add(relation)
                single_word_relations = list(single_word_relations)
                cur_point['relations'] = single_word_relations
            else:
                cur_point[key] = relation_data.loc[index, key]
        data.append(cur_point)
    random.shuffle(data)
    print('data num:', len(data))
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)


def process_dev_data():
    template_path = dir_path + 'relation_templates.json'
    relation_data_path = dir_path + 'dev_with_lstmrels.csv'
    data_path = dir_path + 'infotabs_dev_relations.json'
    template_file = open(template_path, 'r')
    relation_templates = json.load(template_file)
    relation_data = pd.read_csv(relation_data_path)
    relation_data_columns = ['table_id', 'premise', 'hypothesis', 'label', 'Rel_with_toks']
    id_2_label = {0: 'C', 1: 'N', 2: 'E'}
    data = []
    for index in range(len(relation_data)):
        cur_point = {}
        for key in relation_data_columns:
            if key == 'label':
                cur_point[key] = id_2_label[relation_data.loc[index, key]]
            elif key == 'Rel_with_toks':
                single_word_relations = set()
                if relation_data.loc[index, key] == relation_data.loc[index, key]:
                    token_relations = ast.literal_eval(relation_data.loc[index, key])
                    for token_relation in token_relations:
                        for rel_type in token_relation[3]:
                            relation = (token_relation[0][0], token_relation[1][0], relation_templates[rel_type])
                            relation = relation[0] + ' ' + relation[2] + ' ' + relation[1]
                            single_word_relations.add(relation)
                single_word_relations = list(single_word_relations)
                cur_point['relations'] = single_word_relations
            else:
                cur_point[key] = relation_data.loc[index, key]
        data.append(cur_point)
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)


def shuffle_data(data_path, shuffled_data_path):
    data_file = open(data_path, 'r')
    data = json.load(data_file)
    random.shuffle(data)
    with open(shuffled_data_path, 'w') as f:
        json.dump(data, f, indent=4)


def remove_neutral_data(data_path, binary_data_path):
    data_file = open(data_path, 'r')
    data = json.load(data_file)
    binary_data = []
    for cur_point in data:
        if cur_point['label'] != 'N':
            binary_data.append(cur_point)
    print('binary data num:', len(binary_data))
    with open(binary_data_path, 'w') as f:
        json.dump(binary_data, f, indent=4)


if __name__ == '__main__':
    original_train_data_path = dir_path + 'infotabs_train_relations_shuffled.json'
    original_dev_data_path = dir_path + 'infotabs_dev_relations.json'
    train_data_path = dir_path + 'binary_train_relations_shuffled.json'
    dev_data_path = dir_path + 'binary_dev_relations.json'
    set_random_seed(666)
    process_training_data()
    process_dev_data()
    remove_neutral_data(original_train_data_path, train_data_path)
    remove_neutral_data(original_dev_data_path, dev_data_path)
