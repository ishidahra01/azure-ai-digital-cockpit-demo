from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import os
import mlflow
from mlflow.models import infer_signature
import argparse
import sys
import logging
import re
import ast
import json

import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
from datasets import load_dataset
from datetime import datetime

logger = logging.getLogger(__name__)

def log_params_from_dict(config, mlflow_client, parent_key=''):
    """
    Given a dictionary of parameters, logs non-dictionary values to the specified mlflow client.
    Ignores nested dictionaries.

    Args:
        config (dict): The dictionary of parameters to log.
        mlflow_client: The mlflow client to use for logging.
        parent_key (str): Used to prefix keys (for nested logging).
    """
    for key, value in config.items():
        if isinstance(value, dict):
            continue
        elif isinstance(value, list):
            full_key = f"{parent_key}.{key}" if parent_key else key
            mlflow_client.log_param(full_key, ','.join(map(str, value)))
        else:
            full_key = f"{parent_key}.{key}" if parent_key else key
            mlflow_client.log_param(full_key, value)
            

def load_model(args):

    model_name_or_path = args.model_name_or_path    
    model_kwargs = dict(
        revision="main",
		trust_remote_code=True,
		attn_implementation="flash_attention_2",
		torch_dtype=torch.bfloat16,
		use_cache=False,
		device_map="cuda"
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_eos_token=False, add_bos_token=False)
    print(f"tokenizer config: {tokenizer}")
    # tokenizer.model_max_length = args.max_seq_length
    # tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # tokenizer.padding_side = "right"
    return model, tokenizer

## Add hishida
def chat_formatting_fn_calling(example):
    '''
    This function formats the chat data into a format that can be used by the model
    for training and evaluation. The function takes a single example as input and
    returns a dictionary with the following keys:
    - messages: A list of dictionaries with the following keys:
        - role: The role of the message, either "user", "assistant", or "system"
        - content: The content of the message
    '''
    
    return {"messages": [
        {"role": "system", "content": example["conversations"][0]["value"]},
        {"role": "user", "content": example["conversations"][1]["value"]},
        {"role": "assistant", "content": example["conversations"][2]["value"]}
    ]}

def tokenize_function(example, tokenizer):
    '''
    This function tokenizes the chat data for training. The function 
    takes a single example as input, the example must have messages property, 
    and returns a dictionary with the following keys:
    - text: The tokenized input text with the generation prompt
    '''
    example["text"] = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return example

def test_tokenize_function(example, tokenizer):
    '''
    This function tokenizes the chat data and adds the generation prompt to the input
    text. The function takes a single example as input, the example must have messages
    property, and returns a dictionary with the following keys:
    - text: The tokenized input text with the generation prompt
    '''
    example["text"] = tokenizer.apply_chat_template(
        example["messages"][:-1], tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    return example
    # return {"text": tokenizer.apply_chat_template(example["messages"][:-1], tokenize=True, add_generation_prompt=True, padding=True, return_tensors="pt")}

def extract_assistant_text(input_string):
    ''' 
    Regular expression pattern to find text between <|assistant|> and <|end|> tags  
    Sometimes <|end|> is not present, so we need to use the following pattern
    '''
    pattern = re.compile(r'<\|assistant\|>(.*?)(<\|end\|>|<\|endoftext\|>|$)', re.DOTALL)
    # Find all occurrences of the pattern  
    matches = pattern.findall(input_string) 
    return matches

def extract_tool_call(input_string):
    '''
    Extracts the function call from the input string
    '''
    
    try:
        if "```json" in input_string:
            input_string = input_string.split("```json")[1]
        if "```" in input_string:
            input_string = input_string.split("```")[0]
        matches = [tc for tc in input_string.split("</tool_call>") if "<tool_call>\n" in tc]
        matches = [tc.split("<tool_call>\n")[-1] for tc in matches]
        matches = [tc.replace("<tool_call>\n", "").strip() for tc in matches]
        try:
            matches = [json.loads(tc) for tc in matches]
        except:
            matches = [ast.literal_eval(tc) for tc in matches]
        matches = [json.dumps(tc) for tc in matches]
        return matches
    except:
        print(f"Error extracting tool call from: {input_string}")
        return []

## Add hishida
def preprocess_for_finetuning(data, tokenizer):
    example = {}
    # 質問と回答を取得
    question = " ".join(data["question"]) if isinstance(data["question"], list) else data["question"]
    context = " ".join(data["context"]["contexts"]) if isinstance(data["context"]["contexts"], list) else data["context"]["contexts"]
    answer = " ".join(data["long_answer"]) if isinstance(data["long_answer"], list) else data["long_answer"]

    messages = [
        {"role": "system", "content": "You are the SME (Subject Matter Expert) in Medical. Please answer the questions accurately."},
        {"role": "user", "content": context + ' ' + question}, # append context to beginning of Question
        {"role": "assistant", "content": answer}
    ]
    example["messages"] = messages
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    
    return example



def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    # Add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

def main(args):

    ###################
    # Hyper-parameters
    ###################
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key    
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    use_wandb = len(args.wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)

    training_config = {
        "bf16": True,
        "do_eval": False,
        "learning_rate": args.learning_rate,
        "log_level": "info",
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_train_epochs": args.epochs,
        "max_steps": -1,
        "output_dir": "./checkpoint_dir",
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "remove_unused_columns": True,
        "save_steps": args.save_steps,
        "save_total_limit": 1,
        "seed": args.seed,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "gradient_accumulation_steps": args.grad_accum_steps,
        "weight_decay": 0.1,
        "evaluation_strategy": "steps",
        "eval_steps": 50
        # "warmup_ratio": args.warmup_ratio,
    }

    peft_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        #"target_modules": "all-linear",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_to_save": None,
    }

    train_conf = TrainingArguments(
        **training_config,
        report_to="wandb" if use_wandb else "azure_ml",
        run_name=args.wandb_run_name if use_wandb else None,    
    )
    peft_conf = LoraConfig(**peft_config)
    model, tokenizer = load_model(args)
    
    print(tokenizer.model_max_length)
    
    

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_conf.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
        + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_conf}")
    logger.info(f"PEFT parameters {peft_conf}")    

    ##################
    # Data Processing
    ##################
    train_dataset = load_dataset('json', data_files=os.path.join(args.train_dir, 'dataset_train.jsonl'), split='train')
    eval_dataset = load_dataset('json', data_files=os.path.join(args.train_dir, 'dataset_validation.jsonl'), split='train')
    column_names = list(train_dataset.features)
    
    tokenized_train_dataset = train_dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    
    print(tokenized_train_dataset.column_names)
    print(f"Processed train dataset[0] : {tokenized_train_dataset[0]}")
    
    with mlflow.start_run() as run:     
        
        log_params_from_dict(training_config, mlflow)
        log_params_from_dict(peft_config, mlflow)
        
        ###########
        # Training
        ###########
        trainer = SFTTrainer(
            model=model,
            args=train_conf,
            peft_config=peft_conf,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            tokenizer=tokenizer,
            # packing=True,
        )

        # Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer.train()

        #############
        # Logging
        #############
        metrics = trainer_stats.metrics

        # Show final memory and time stats 
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_lora/max_memory*100, 3)

        logger.info(f"{metrics['train_runtime']} seconds used for training.")
        logger.info(f"{round(metrics['train_runtime']/60, 2)} minutes used for training.")
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
                
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        trainer.save_state()
        
        #############
        # Evaluation
        #############
        # tokenizer.padding_side = "left"
        # metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(processed_eval_dataset)
        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

        # ############
        # # Save model
        # ############
        os.makedirs(args.model_dir, exist_ok=True)

        if args.save_merged_model:
            model_tmp_dir = "model_tmp"
            os.makedirs(model_tmp_dir, exist_ok=True)
            trainer.model.save_pretrained(model_tmp_dir)
            print(f"Save merged model: {args.model_dir}")
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(model_tmp_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(args.model_dir, safe_serialization=True)
        else:
            print(f"Save PEFT model: {args.model_dir}")    
            trainer.model.save_pretrained(args.model_dir)

        tokenizer.save_pretrained(args.model_dir)             


def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()
    # curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # hyperparameters
    parser.add_argument("--model_name_or_path", default="microsoft/Phi-3.5-mini-instruct", type=str, help="Input directory for training")    
    parser.add_argument("--train_dir", default="data", type=str, help="Input directory for training")
    parser.add_argument("--model_dir", default="./model", type=str, help="output directory for model")
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--train_batch_size", default=8, type=int, help="training - mini batch size for each gpu/process")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="evaluation - mini batch size for each gpu/process")
    parser.add_argument("--learning_rate", default=5e-06, type=float, help="learning rate")
    parser.add_argument("--logging_steps", default=2, type=int, help="logging steps")
    parser.add_argument("--save_steps", default=100, type=int, help="save steps")    
    parser.add_argument("--grad_accum_steps", default=4, type=int, help="gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="warmup ratio")
    parser.add_argument("--max_seq_length", default=2048, type=int, help="max seq length")
    parser.add_argument("--save_merged_model", type=bool, default=False)

    # lora hyperparameters
    parser.add_argument("--lora_r", default=16, type=int, help="lora r")
    parser.add_argument("--lora_alpha", default=16, type=int, help="lora alpha")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="lora dropout")
    
    # wandb params
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_watch", type=str, default="gradients") # options: false | gradients | all
    parser.add_argument("--wandb_log_model", type=str, default="false") # options: false | true

    # parse args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    #sys.argv = ['']
    args = parse_args()
    main(args)
