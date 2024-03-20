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


def get_dataframes(filename):
    '''
    Returns the full, train, and test dataframes
    '''
    full_df = build_dataset_from_file(filename)
    prompt_df = create_prompt_dataframe(full_df)
    dataset = Dataset(pa.Table.from_pandas(prompt_df))
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=12)
    test_data_df = pd.DataFrame(test_data)
    train_data_df = pd.DataFrame(train_data)

    return prompt_df, train_data_df, test_data_df

if name=="__main__":
    filename = '../content/CT_Protocol/data/dataset031524.xlsx'
    _, _, test_df = get_dataframes(filename)
    token = userdata.get('HUGGINGFACE_TOKEN')
    api = HfApi(token=token)
    base_model = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer = tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    base_model_score = test_model(test_df, pipe)
    print('Base model score:', base_model_score)



