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
from dotenv import load_dotenv
from utilities import *

# create and save train_data_df and the training dataset
def create_train_data_df(train_data, prompt_instruction = prompt_instruction2):
  '''
  Create the training dataset in the format required for the model
  Input: train_data: a list of dictionaries
  Input: prompt_instruction: a string
  Output: train_data_df: a dataframe
  '''
  train_data_df = pd.DataFrame(train_data)
  maker_df = train_data_df.copy()
  for index, row in maker_df.iterrows():
    maker_df.loc[index, 'text'] = f"""<s>[INST] {prompt_instruction}{row['text']} [/INST] \\n {row['labels']} </s>"""
    maker_df.loc[index, 'labels'] = row['labels']

  maker_df.head()
  maker_df.drop(columns=['prompt_question_json', '__index_level_0__'], inplace=True)
  #train_dataset = Dataset.from_pandas(maker_df)
  train_dataset = Dataset(pa.Table.from_pandas(maker_df))

  return train_dataset



def main():
    
    filename = '../content/CT_Protocol/data/dataset031524.xlsx'
    _, train_data, _ = get_dataframes(filename)
    train_dataset = create_train_data_df(train_data)
    save_model_name = 'mille055/auto_protocol'

    
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')
    write_token = os.getenv('HUGGINFACE_WRITE_TOKEN')
    
    api = HfApi(token=token)
    base_model = "mistralai/Mistral-7B-Instruct-v0.2"


    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )


    model.config.use_cache = False # silence the warnings.
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.bos_token, tokenizer.eos_token



    # Ensure to clear cache if anything is not used
    torch.cuda.empty_cache()

    #Adding the adapters in the layers
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
    model = get_peft_model(model, peft_config)


    # Setting hyperparameters
    training_arguments = TrainingArguments(
        output_dir="/content/CT_Protocol/data",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )
    # Setting sft parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length= 4000,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing= False,
    )

    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer = tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Training the model
    trainer.train()


    # Save the fine-tuned model
    trainer.model.save_pretrained(new_model)
    model.config.use_cache = True

    trainer.model.push_to_hub(save_model_name, token=write_token)
    tokenizer.push_to_hub(save_model_name, token=write_token)
    
if __name__ =="__main__":
    main()

    
    



