import json


def process_tempquestions():
    dir_path = '/path/to/working/dir/Temporal/'
    input_file = dir_path + 'TempQuestions.json'
    output_file = dir_path + 'implicit_temporal_questions.json'
    remaining_file = dir_path + 'remaining_temporal_questions.json'
    temp_data = json.load(open(input_file, 'r'))
    implicit_data = []
    remaining_data = []
    implicit_qa_num = 0
    remaining_qa_num = 0
    explicit_qa_num = 0
    ordinal_qa_num = 0
    tempans_qa_num = 0
    qa_num = 0
    multiple_ans_num = 0
    for qa in temp_data:
        qa_num += 1
        for qa_type in qa['Type']:
            if qa_type not in ['Implicit', 'Explicit', 'Ordinal', 'Temp.Ans']:
                print(qa_type)
        if 'Implicit' in qa['Type'] or 'Implicit:' in qa['Type']:
            if len(qa['Gold answer']) == 1:
                implicit_qa_num += 1
                implicit_data.append(qa)
            else:
                multiple_ans_num += 1
        else:
            if len(qa['Gold answer']) == 1:
                remaining_qa_num += 1
                remaining_data.append(qa)
            else:
                multiple_ans_num += 1
        if 'Explicit' in qa['Type']:
            explicit_qa_num += 1
        if 'Ordinal' in qa['Type']:
            ordinal_qa_num += 1
        if 'Temp.Ans' in qa['Type']:
            tempans_qa_num += 1
    print('qa num:', qa_num)
    print('explicit qa num:', explicit_qa_num)
    print('ordinal qa num:', ordinal_qa_num)
    print('tempans qa num:', tempans_qa_num)
    print('multiple ans num:', multiple_ans_num)
    print('kept implicit qa num:', implicit_qa_num)
    print('kept remaining qa num:', remaining_qa_num)
    with open(output_file, 'w') as f:
        json.dump(implicit_data, f, indent=4)
    with open(remaining_file, 'w') as f:
        json.dump(remaining_data, f, indent=4)


if __name__ == '__main__':
    process_tempquestions()



