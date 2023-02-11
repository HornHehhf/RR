# RR
This is the code repository for the Arxiv paper [Rethinking with Retrieval: Faithful Large Language Model Inference](https://arxiv.org/pdf/2301.00303.pdf).
If you use this code for your work, please cite
```
@article{he2022rethinking,
  title={Rethinking with Retrieval: Faithful Large Language Model Inference},
  author={He, Hangfeng and Zhang, Hongming and Roth, Dan},
  journal={arXiv preprint arXiv:2301.00303},
  year={2022}
}
```


## Installing dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python==3.7.10\
pip install -r requirements.txt

If you couldn't install all dependencies in one virtual environment, you may need three different virtual environments:
- pyserini (pyserini==0.19.2): for BM25 in commonsense_wikipedia.py
- transformers (transformers==4.23.1, sentence_transformers==2.2.2): for huggingface transformer models in commonsense_evidence.py, temporal_evidence.py, tabular_evidence.py
- reasoning: default environment dependencies in the remaining files

## Code organization
The code is organized as follows:
- commonsense (commonsense reasoning)
    - commonsense_gpt3.py (chain-of-thought prompting with GPT-3)
    - commonsense_wikipedia.py (retrieve knowledge from wikipedia)
    - commonsense_evidence.py (supporting evidence from retrieved knowledge for inference)
    - commonsense_inference.py (faithful inference)
    - commonsense_utils.py (utility functions)
- temporal (temporal reasoning)
    - temporal_preprocessing.py (data preprocessing)
    - temporal_gpt3.py (chain-of-thought prompting with GPT-3)
    - temporal_wikidata.py (retrieve knowledge from wikidata)
    - temporal_evidence.py (supporting evidence from retrieved knowledge for inference)
    - temporal_inference.py (faithful inference)
    - temporal_utils.py (utility functions)
- tabular (tabular reasoning)
    - tabular_preprocessing.py (data preprocessing)
    - tabular_gpt3.py (chain-of-thought prompting with GPT-3)
    - tabular_evidence.py (supporting evidence from retrieved knowledge for inference)
    - tabular_inference.py (faithful inference)
    - tabular_utils.py (utility functions)
    - **relation_templates.json (relation templates; needs to be put under /path/to/working/dir/Tabular/)**


## Change the working path
Change the /path/to/working/dir to the path to your working directory.

## Export OPENAI API KEY
You need to export your own OpenAI API key before running experiments with [OpenAI API](https://openai.com/api/), i.e., export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY

## Data preparation 
Download and put [StrategyQA dataset](https://github.com/eladsegal/strategyqa/tree/main/data/strategyqa) under /path/to/working/dir/Commonsense/\
Download and put [TempQeustions datset](http://qa.mpi-inf.mpg.de/TempQuestions.zip) under /path/to/working/dir/Temporal/\
Download and put [INFOTABS dataset](https://github.com/Varun221/trans-kblstm/tree/master/data/kinfotabs_withrels) under /path/to/working/dir/Tabular/

## Reproducing experiments
To reproduce the experiments for commonsense reasoning:
```
python commonsense_gpt3.py
python commonsense_wikipedia.py
python commonsense_evidence.py
python commonsense_inference.py
```
To reproduce the experiments for temporal reasoning:
```
python temporal_preprocessing.py
python temporal_gpt3.py
python temporal_wikidata.py
python temporal_evidence.py
python temporal_inference.py
```
To reproduce the experiments for tabular reasoning:
```
python tabular_preprocessing.py
python tabular_gpt3.py
python tabular_evidence.py
python tabular_inference.py
```