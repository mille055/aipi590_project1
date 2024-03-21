### Code adapted from https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, LlamaTokenizer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,re
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from google.colab import userdata
import json
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from utilities import *
from dotenv import load_dotenv

def main():
    filename = '../content/CT_Protocol/data/dataset031524.xlsx'
    _, _, test_df = get_dataframes(filename)
    model_path = "mille055/auto_protocol2"

    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')
    api = HfApi(token=token)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, framework="pt")

    ft_model_score = test_model2(test_df, pipe=pipe)
    print(ft_model_score)   


if __name__ == "__main__":
    main()