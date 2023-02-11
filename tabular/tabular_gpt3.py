import os
import json
import openai
import time

from tabular_utils import evaluate_qa

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

ori_prompt = "Charles Sumner Tainter was Born on April 25, 1854   ( 1854-04-25 )   Watertown, Massachusetts, U.S..  Charles Sumner Tainter was Died on April 20, 1940   ( 1940-04-21 )  (aged 85)  San Diego, California, U.S..  The Nationality of Charles Sumner Tainter are American.  The Known for of Charles Sumner Tainter are Photophone, phonograph Father Of The Speaking Machine.\n" \
			 "Question: Charles Sumner Tainter never left the state of Massachusetts. True or False?\n" \
			 "Answer: False.\n\n" \
             "The Region of Curitiba are South.  The Elevation of Curitiba are 934.6 m (3,066.3 ft).  The Density of Curitiba are 4,062/km 2  (10,523/sq mi).  The Metro density of Curitiba are 210.9/km 2  (546.2/sq mi).\n" \
			 "Question: Curitiba is above sea level. True or False?\n" \
			 "Answer: True.\n\n" \
             "Charles (Prince of Wales) was Born on 14 November 1948 ( 1948-11-14 )   (age 70)  Buckingham Palace, London, England.  The Spouse of Charles (Prince of Wales) are Lady Diana Spencer ( m.   1981 ;  div.   1996 )  , and Camilla Parker Bowles  ( m.   2005 ).  The Issue of Charles (Prince of Wales) are Prince William, Duke of Cambridge , and Prince Harry, Duke of Sussex.\n" \
			 "Question: Charles was born in 1948 and has been married twice. True or False?\n" \
			 "Answer: True.\n\n" \
             "The Born of Idris Elba are 6 September 1972  (age 46)   Hackney, London, England.  The Residence of Idris Elba are London.  The Other names of Idris Elba are DJ Big Driis, Big Driis the Londoner, Big Driis, and 7 Dub.  The Occupation of Idris Elba are Actor, producer, director, musician, and DJ.\n" \
			 "Question: Idris Elba is an English entertainer. True or False?\n" \
			 "Answer: True.\n\n" \
             "The Breed of Jean, the Vitagraph Dog are Scotch Collie.  The Sex of Jean, the Vitagraph Dog are Female.  The Born of Jean, the Vitagraph Dog are 1902 Eastport, Maine.  The Years active of Jean, the Vitagraph Dog are 1909 - 1916.\n"\
             "Question: Jean, the Vitagraph Dog was a Golden Retriever which perform in circus. True or False?\n"\
             "Answer: False.\n\n"\
             "The Studio of Hydrograd are Sphere Studios, North Hollywood, Los Angeles.  The Genre of Hydrograd are Hard rock.  The Label of Hydrograd are Roadrunner.  The Producer of Hydrograd are Jay Ruston.\n" \
             "Question: Hydrograd is in the rap genre. True or False?\n" \
             "Answer: False.\n\n" \

chain_prompt = "Charles Sumner Tainter was Born on April 25, 1854   ( 1854-04-25 )   Watertown, Massachusetts, U.S..  Charles Sumner Tainter was Died on April 20, 1940   ( 1940-04-21 )  (aged 85)  San Diego, California, U.S..  The Nationality of Charles Sumner Tainter are American.  The Known for of Charles Sumner Tainter are Photophone, phonograph Father Of The Speaking Machine.\n" \
			   "Question: Charles Sumner Tainter never left the state of Massachusetts. True or False?\n" \
			   "Answer: Charles Sumner Tainter was died in San Diego, California, U.S.. California is a state. Thus, Charles Sumner Tainter has left the state of Massachusetts. So the answer is false.\n\n" \
               "The Region of Curitiba are South.  The Elevation of Curitiba are 934.6 m (3,066.3 ft).  The Density of Curitiba are 4,062/km 2  (10,523/sq mi).  The Metro density of Curitiba are 210.9/km 2  (546.2/sq mi).\n" \
			   "Question: Curitiba is above sea level. True or False?\n" \
			   "Answer: The elevation of Curitiba are 934.6 m (3,066.3 ft). Elevation is a hypernym of level. Thus, Curitiba is above sea level. So the answer is true.\n\n" \
               "Charles (Prince of Wales) was Born on 14 November 1948 ( 1948-11-14 )   (age 70)  Buckingham Palace, London, England.  The Spouse of Charles (Prince of Wales) are Lady Diana Spencer ( m.   1981 ;  div.   1996 )  , and Camilla Parker Bowles  ( m.   2005 ).  The Issue of Charles (Prince of Wales) are Prince William, Duke of Cambridge , and Prince Harry, Duke of Sussex.\n" \
			   "Question: Charles was born in 1948 and has been married twice. True or False?\n" \
			   "Answer: Charles (Prince of Wales) was Born on 14 November 1948. The Spouse of Charles (Prince of Wales) are Lady Diana Spencer ( m.   1981 ;  div.   1996 )  , and Camilla Parker Bowles  ( m.   2005 ). Married is related to spouse. Thus, Charles was born in 1948 and has been married twice. So the answer is true.\n\n" \
               "The Born of Idris Elba are 6 September 1972  (age 46)   Hackney, London, England.  The Residence of Idris Elba are London.  The Other names of Idris Elba are DJ Big Driis, Big Driis the Londoner, Big Driis, and 7 Dub.  The Occupation of Idris Elba are Actor, producer, director, musician, and DJ.\n" \
			   "Question: Idris Elba is an English entertainer. True or False?\n" \
			   "Answer: The residence of Idris Elba is London. English is related to London. The occupation of Idris Elba are actor, producer, director, musician, and DJ. Actor is a hyponym of entertainer. Musician is a hyponym of entertainer. DJ is an entertainer. Thus, Idris Elba is an English entertainer. So the answer is true.\n\n" \
               "The Breed of Jean, the Vitagraph Dog are Scotch Collie.  The Sex of Jean, the Vitagraph Dog are Female.  The Born of Jean, the Vitagraph Dog are 1902 Eastport, Maine.  The Years active of Jean, the Vitagraph Dog are 1909 - 1916.\n"\
               "Question: Jean, the Vitagraph Dog was a Golden Retriever which perform in circus. True or False?\n"\
               "Answer: The Breed of Jean, the Vitagraph Dog are Scotch Collie. Collie is a hyponym of dog. Retriever is a hyponym of dog. Thus, Jean, the Vitagraph Dog was not a Golden Retriever which perform in circus. So the answer is false.\n\n"\
               "The Studio of Hydrograd are Sphere Studios, North Hollywood, Los Angeles.  The Genre of Hydrograd are Hard rock.  The Label of Hydrograd are Roadrunner.  The Producer of Hydrograd are Jay Ruston.\n" \
               "Question: Hydrograd is in the rap genre. True or False?\n" \
               "Answer: The Genre of Hydrograd are Hard rock. Rap is distinct from rock. Thus, Hydrograd is not in the rap genre. So the answer is false.\n\n" \


def run_gpt3(test_path, output_path, output_option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        premise = test_case['premise']
        hypothesis = test_case['hypothesis']
        if output_option == 'zero_shot_gpt3':
            combined_prompt = premise + "\nQuestion: " + hypothesis + " True or False?\nAnswer:"
        elif output_option == 'few_shot_gpt3':
            combined_prompt = ori_prompt + premise + "\nQuestion: " + hypothesis + " True or False?\nAnswer:"
        elif output_option == 'chain_of_thought_gpt3':
            combined_prompt = chain_prompt + premise + "\nQuestion: " + hypothesis + " True or False?\nAnswer:"
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
        premise = test_case['premise']
        hypothesis = test_case['hypothesis']
        combined_prompt = chain_prompt + premise + "\nQuestion: " + hypothesis + " True or False?\nAnswer:"
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
    dir_path = '/path/to/working/dir/Tabular/'
    test_path = dir_path + 'binary_dev_relations.json'
    zero_shot_gpt3_path = dir_path + 'GPT-3/binary_dev_zero_shot_gpt3.json'
    few_shot_gpt3_path = dir_path + 'GPT-3/binary_dev_few_shot_gpt3.json'
    chain_of_thought_gpt3_path = dir_path + 'GPT-3/binary_dev_chain_of_thought_gpt3.json'
    self_consistency_gpt3_path = dir_path + 'GPT-3/binary_dev_self_consistency_gpt3.json'
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

