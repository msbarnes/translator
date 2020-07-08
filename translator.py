import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
import os
import argparse
from dotenv import load_dotenv
import logging
from typing import List

logging.basicConfig(level=logging.INFO)

load_dotenv()
IMDB_URL=os.getenv("IMDB_URL")
#"https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv"

parser = argparse.ArgumentParser(
    description='input source and target language to translate a list of text',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--source", type=str, help="source language code")
parser.add_argument("--target", type=str, help="target language code")

def load_model(route:str):
    """loads source-target NMT model"""

    model = f'opus-mt-{route}'
    path = os.path.join('data', model)
    try:
        model = MarianMTModel.from_pretrained(path)
        tokenizer = MarianTokenizer.from_pretrained(path)
    except:
        return f"make sure you have downloaded model for {route} translation"
    return model, tokenizer

def translate(model, tokenizer, text:List[str]) -> List[str]:
    """

    """
    
    model, tokenizer = model, tokenizer
    batch = tokenizer.prepare_translation_batch(src_texts=text)
    gen = model.generate(**batch)
    translated_text = tokenizer.batch_decode(gen, skip_special_tokens=True)
    
    return translated_text

def main():

    #TODO: suppress W&B login message

    args = parser.parse_args()
    route = f"{args.source}-{args.target}"
    df = pd.read_csv(IMDB_URL, delimiter='\t', header=None)
    df.columns = ['text', 'label']

    #limit to first 50 reviews for quick runs
    df2 = df.iloc[:50,:]
    text = df2['text'].to_list()

    logging.info("Loading model and tokenizer")
    model, tokenizer = load_model(route)

    logging.info("Loading complete. Translating text...")

    #TODO: add tqdm to monitor progress
    translated_text = translate(model, tokenizer, text)

    df2['translated'] = pd.Series(translated_text)

    logging.info("Translation complete. Saving to csv")

    df2.to_csv("translated_imdb_reviews.csv", index=False)


if __name__ == "__main__":
    main()