from pyserini.search.lucene import LuceneSearcher
import json
import spacy
import time

nlp = spacy.load("en_core_web_md")
ssearcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr')


def information_retrieval(query):
    hits = ssearcher.search(query, 10)
    # for i in range(0, 10):
    #     print(f'{i + 1:2} {hits[i].docid:7} {hits[i].score:.5f}')
    paragraphs = []
    for i in range(len(hits)):
        doc = ssearcher.doc(hits[i].docid)
        json_doc = json.loads(doc.raw())
        paragraphs.append(json_doc['contents'])
    return paragraphs


def get_wikipages(test_path, output_path):
    output_option = 'sparse_retrieval'
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        predictions = test_case['self_consistency_gpt3']
        test_case[output_option] = []
        for prediction in predictions:
            prediction = str(prediction).strip()
            nlp_text = nlp(prediction).sents
            sent_pages = []
            for sent in nlp_text:
                sent = str(sent)
                paragraphs = information_retrieval(sent)
                sent_pages.append((sent, paragraphs))
            test_case[output_option].append(sent_pages)
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=4)


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/Commonsense/'
    self_consistency_gpt3_path = dir_path + 'GPT-3/strategyqa_dev_self_consistency_gpt3.json'
    self_consistency_pages_path = dir_path + 'GPT-3/strategyqa_dev_self_consistency_gpt3_paragraphs.json'
    time_start = time.time()
    print('retrieve wiki paragraphs')
    get_wikipages(self_consistency_gpt3_path, self_consistency_pages_path)
    time_end = time.time()
    print('time:', time_end - time_start)

