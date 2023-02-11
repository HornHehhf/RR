import os
import json
import time
import openai

from commonsense_utils import evaluate_qa

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

ori_prompt = "Q: Do hamsters provide food for any animals?\n" \
             "A: Yes.\n\n" \
             "Q: Could Brooke Shields succeed at University of Pennsylvania?\n" \
			 "A: Yes.\n\n" \
			 "Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?\n" \
			 "A: No.\n\n" \
			 "Q: Yes or no: Is it common to see frost during some college commencements?\n" \
			 "A: Yes.\n\n" \
			 "Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?\n" \
			 "A: No.\n\n" \
			 "Q: Yes or no: Would a pear sink in water?\n" \
			 "A: No.\n\n" \

chain_prompt = "Q: Do hamsters provide food for any animals?\n" \
               "A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.\n\n" \
               "Q: Could Brooke Shields succeed at University of Pennsylvania?\n" \
			   "A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.\n\n" \
			   "Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?\n" \
			   "A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is no.\n\n" \
			   "Q: Yes or no: Is it common to see frost during some college commencements?\n" \
			   "A: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.\n\n" \
			   "Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?\n" \
			   "A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.\n\n" \
			   "Q: Yes or no: Would a pear sink in water?\n" \
			   "A: The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no.\n\n" \


def run_gpt3(test_path, output_path, output_option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        question = test_case['question']
        if output_option == 'zero_shot_gpt3':
            combined_prompt = "Q: Yes or no: " + question + "\nA: So the answer is "
        elif output_option == 'few_shot_gpt3':
            combined_prompt = ori_prompt + "Q: Yes or no: " + question + "\nA:"
        elif output_option == 'chain_of_thought_gpt3':
            combined_prompt = chain_prompt + "Q: Yes or no: " + question + "\nA:"
        print(combined_prompt)
        response = openai.Completion.create(engine="text-davinci-002", prompt=combined_prompt, temperature=0,
											max_tokens=256)
        print(response)
        test_case[output_option] = response['choices'][0]['text']
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=4)


def run_gpt3_multiple(test_path, output_path, output_option):
    self_consistency_rounds = 10
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        question = test_case['question']
        combined_prompt = chain_prompt + "Q: Yes or no: " + question + "\nA:"
        print(combined_prompt)
        for i in range(self_consistency_rounds):
            response = openai.Completion.create(engine="text-davinci-002", prompt=combined_prompt, temperature=0.7,
                                                max_tokens=256)
            print(response)
            if output_option not in test_case:
                test_case[output_option] = []
            test_case[output_option].append(response['choices'][0]['text'])
            with open(output_path, 'w') as f:
                json.dump(test_data, f, indent=4)


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/Commonsense/'
    test_path = dir_path + 'dev.json'
    zero_shot_gpt3_path = dir_path + 'GPT-3/strategyqa_dev_zero_shot_gpt3.json'
    few_shot_gpt3_path = dir_path + 'GPT-3/strategyqa_dev_few_shot_gpt3.json'
    chain_of_thought_gpt3_path = dir_path + 'GPT-3/strategyqa_dev_chain_of_thought_gpt3.json'
    self_consistency_gpt3_path = dir_path + 'GPT-3/strategyqa_dev_self_consistency_gpt3.json'
    time_start = time.time()
    run_gpt3(test_path, zero_shot_gpt3_path, output_option='zero_shot_gpt3')
    evaluate_qa(zero_shot_gpt3_path, option='zero_shot_gpt3')
    run_gpt3(test_path, few_shot_gpt3_path, output_option='few_shot_gpt3')
    evaluate_qa(few_shot_gpt3_path, option='few_shot_gpt3')
    run_gpt3(test_path, chain_of_thought_gpt3_path, output_option='chain_of_thought_gpt3')
    evaluate_qa(chain_of_thought_gpt3_path, option='chain_of_thought_gpt3')
    run_gpt3_multiple(test_path, self_consistency_gpt3_path, output_option='self_consistency_gpt3')
    evaluate_qa(self_consistency_gpt3_path, option='self_consistency_gpt3')
    time_end = time.time()
    print('time:', time_end - time_start)
