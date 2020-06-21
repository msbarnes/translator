import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
import os
from dotenv import load_dotenv
import logging

load_dotenv()
IMDB_URL=os.getenv("IMDB_URL")
#"https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv"


def load_model(route):
    model = f'opus-mt-{route}'
    path = os.path.join('data',model)
    try:
        model = MarianMTModel.from_pretrained(path)
        tokenizer = MarianTokenizer.from_pretrained(path)
    except:
        return f"Make sure you have downloaded model for {route} translation"
    return model, tokenizer

df = pd.read_csv(IMDB_URL, delimiter='\t', header=None)
text = df[0].to_list()

model, tokenizer = load_model('en-fr')

