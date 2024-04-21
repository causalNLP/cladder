## python file that generates predictions from a pretrained language model from huggingface transformers
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer
import csv
import pandas as pd

def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def convert_to_norm(value):
    prefix2norm = {
        'Yes': 1,
        'No': 0,
    }
    invalid = -1
    value = str(value).lower().strip().strip('"')

    for prefix, norm in prefix2norm.items():
        if value.startswith(prefix.lower()):
            return norm
    return invalid

def main(locationLlamaHF,outputFileName,inputFileName):
    tokenizer = AutoTokenizer.from_pretrained(locationLlamaHF,cache_dir="~/cache/")
    model = AutoModelForCausalLM.from_pretrained(locationLlamaHF,device_map="auto",cache_dir="~/cache/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## read prompts from csv and generate predictions
    df=pd.read_csv(inputFileName)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    with open(outputFileName, 'w', newline='') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerow(['question_id', 'rung', 'prompt', 'truth', 'truth_norm', 'pred', 'pred_norm'])  # Include 'question_id' column in the header

    with torch.no_grad():
        for i in range(df.shape[0]):
            prompt = df.at[i, 'prompt']
            prompt=generate_prompt(prompt)
            question_id = df.at[i, 'question_id']  # Fetch question_id from 'question_id' column
            truth = df.at[i, 'truth']
            rung = df.at[i, 'rung']
            truth_norm = df.at[i, 'truth_norm']
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=10,
                temperature=0
            )
            outputs = tokenizer.decode(output_sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred_norm = convert_to_norm(outputs)
            with open(outputFileName, 'a', newline='') as csvoutput:
                writer = csv.writer(csvoutput)
                writer.writerow([question_id] + [rung] + [prompt] + [truth] + [truth_norm] + [outputs] + [pred_norm])

if __name__ == '__main__':
    ## model location HuggingFace format
    locationLlamaHF="./alpaca-7b"
    outputFileName="./alpaca007_causal_benchmark.csv"
    inputFileName="./causal_benchmark_data_llama.csv"
    main(locationLlamaHF,outputFileName,inputFileName)
