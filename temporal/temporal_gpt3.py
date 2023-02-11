import os
import json
import openai
import time

from temporal_utils import evaluate_qa

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

ori_prompt = "Q: who was governor of minnesota when maathaad maathaadu mallige was released?\n" \
             "A: The answer is Tim Pawlenty.\n\n" \
             "Q: who was us president during the costa rican civil war?\n" \
			 "A: The answer is Harry S. Truman.\n\n" \
			 "Q: who was governor of oregon when the collector was released?\n" \
			 "A: The answre is Mark Hatfield.\n\n" \
			 "Q: who was governor of oregon when shanghai noon was released?\n" \
			 "A: The answer is John Kitzhaber.\n\n" \
			 "Q: who was us president when john andrew shulze was a teenager?\n" \
			 "A: The answer is George Washington.\n\n" \
			 "Q: who was us president during the seventh coalition?\n" \
			 "A: The answer is James Madison.\n\n" \

chain_prompt = "Q: who was governor of minnesota when maathaad maathaadu mallige was released?\n" \
               "A: Maathaad Maathaadu Mallige was released on 24 August 2007. Tim Pawlenty served as the 39th governor of Minnesota from 2003 to 2011. Thus, Tim Pawlenty was governor of minnesota when maathaad maathaadu mallige was released. So the answer is Tim Pawlenty.\n\n" \
               "Q: who was us president during the costa rican civil war?\n" \
			   "A: The Costa Rican civil war was a civil war in Costa Rica from 12 March to 24 April 1948. Harry S. Truman was the 33rd president of the United States, serving from 1945 to 1953. Thus, Harry S. Truman was us president during the costa rican civil war. So the answer is Harry S. Truman.\n\n" \
			   "Q: who was governor of oregon when the collector was released?\n" \
			   "A: The Collector premiered at the Cannes Film Festival on May 20, 1965. Mark Hatfield served as the 29th governor of Oregon from 1959 to 1967. Thus, Mark Hatfield was governor of oregon when the collector was released. So the answer is Mark Hatfield.\n\n" \
			   "Q: who was governor of oregon when shanghai noon was released?\n" \
			   "A: Shanghai Noon was released on May 26, 2000. John Kitzhaber served as the 35th governor of Oregon from 1995 to 2003. Thus, John Kitzhaber was governor of oregon when shanghai noon was released. So the answer is John Kitzhaber.\n\n" \
			   "Q: who was us president when john andrew shulze was a teenager?\n" \
			   "A: John Andrew Shulze was born on July 19, 1775. A teenager is someone who is between 13 and 19 years old. George Washington served as the first president of the United States from 1789 to 1797. Thus, George Washington was us president when john andrew shulze was a teenager. So the answer is George Washington.\n\n" \
			   "Q: who was us president during the seventh coalition?\n" \
			   "A: The War of the Seventh Coalition was from 20 March to 8 July 1815. James Madison served as the fourth president of the United States from 1809 to 1817. Thus, James Madison was us president during the seventh coalition. So the answer is James Madison.\n\n" \



def run_gpt3(test_path, output_path, output_option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        question = test_case['Question']
        if output_option == 'zero_shot_gpt3':
            combined_prompt = "Q: " + question + "\nA: The answer is "
        elif output_option == 'few_shot_gpt3':
            combined_prompt = ori_prompt + "Q: " + question + "\nA: The answer is "
        elif output_option == 'chain_of_thought_gpt3':
            combined_prompt = chain_prompt + "Q: " + question + "\nA: "
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
        question = test_case['Question']
        combined_prompt = chain_prompt + "Q: " + question + "\nA: "
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
    dir_path = '/path/to/working/dir/Temporal/'
    test_path = dir_path + 'implicit_temporal_questions.json'
    zero_shot_gpt3_path = dir_path + 'GPT-3/implicit_temporal_zero_shot_gpt3.json'
    few_shot_gpt3_path = dir_path + 'GPT-3/implicit_temporal_few_shot_gpt3.json'
    chain_of_thought_gpt3_path = dir_path + 'GPT-3/implicit_temporal_chain_of_thought_gpt3.json'
    self_consistency_gpt3_path = dir_path + 'GPT-3/implicit_temporal_self_consistency_gpt3.json'
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

