import os
from dotenv import load_dotenv
import argparse
import logging
import urllib
from urllib.request import urlretrieve

load_dotenv()

logging.basicConfig(level=logging.INFO)

HUGGINGFACE_S3_BASE_URL=os.getenv("HUGGINGFACE_S3_BASE_URL")
CONFIG=os.getenv("CONFIG")
MODEL=os.getenv("MODEL")
SOURCE=os.getenv("SOURCE")
TARGET=os.getenv("TARGET")
TOKEN=os.getenv("TOKEN")
VOCAB=os.getenv("VOCAB")
FILENAMES=[CONFIG,MODEL,SOURCE,TARGET,TOKEN,VOCAB] 
MODEL_PATH = os.getenv("MODEL_PATH")


parser = argparse.ArgumentParser(
    description='download models by inputting source language and target language',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--source", type=str, help="source language code")
parser.add_argument("--target", type=str, help="target language code")

def download_language_model(source, target):
    """retrieves huggingface NMT source lang to target lang files"""
    model = f"opus-mt-{source}-{target}"
    logging.info(f"Downloading {source} to {target} model...")
    os.makedirs(os.path.join("data", model))
    for f in FILENAMES:
        try:
            print(os.path.join(HUGGINGFACE_S3_BASE_URL, model, f))
            urlretrieve(
                "/".join([HUGGINGFACE_S3_BASE_URL, model, f]),
                os.path.join(MODEL_PATH, model, f),
            )
            logging.info("download complete!")
        except urllib.error.HTTPError:
            logging.info("Error retrieving model from url. Please confirm model exists.")
            os.rmdir(os.path.join("data", model))
            break

if __name__ == "__main__":
    args = parser.parse_args()
    download_language_model(args.source, args.target)