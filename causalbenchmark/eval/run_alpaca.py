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


def main(locationLlamaHF,outputFileName,inputFileName):
    tokenizer = AutoTokenizer.from_pretrained(locationLlamaHF,cache_dir="~/cache/")
    model = AutoModelForCausalLM.from_pretrained(locationLlamaHF,device_map="auto",cache_dir="~/cache/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## read prompts from csv and generate predictions
    df=pd.read_csv(inputFileName)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    with open(outputFileName, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        row=['pred']
        writer.writerow(row)

    with torch.no_grad():
        for i in range(0,df.shape[0],1):
            prompts=list(df['prompt'].values)[i:i+1]
            ## generate prompt in the format as in alpaca repo
            prompts=generate_prompt(prompts[0])
            inputs = tokenizer([prompts], return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=10,temperature=0
            )
            outputs=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            outputs=[[el] for el in outputs]
            with open(outputFileName, 'a') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                writer.writerows(outputs)

if __name__ == '__main__':
    ## model location HuggingFace format
    locationLlamaHF="./alpaca-7b"
    outputFileName="./alpaca007_causal_benchmark.csv"
    inputFileName="./causal_benchmark_data.csv"
    main(locationLlamaHF,outputFileName,inputFileName)
