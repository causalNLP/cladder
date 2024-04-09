"""
Script that generates prompts for the run_llama.py script. The prompts are generated from a JSON file containing the data. The script reads the data from the JSON file, formats the prompts, and saves the prompts to a CSV file.
How to run this script:
python generate_data_llama.py ../../data/test-generate-easy.json
The json file should contain the following keys:
- background: str
- given_info: str
- question: str
The data files are in the /data folder.
"""
import pandas as pd
import argparse
import json

SYSTEM_PROMPT='''You are an expert in causal inference. The following question is not a typical commonsense query, but rather a meticulously designed question created by a professor specializing in causal inference, intended to assess the students' mastery of the course content.'''

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate data script')
    parser.add_argument('input_file', type=str, help='Input JSON file')
    return parser.parse_args()

def read_data(input_file):
    """
    Read data from input JSON file

    Parameters
    ----------
    input_file : str
        Input JSON file containing the data

    Returns
    -------
    dict
        Dictionary containing the data
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def format_prompt(datum, system_prompt):
    """
    Format the prompt for the given data

    Parameters
    ----------
    datum : dict
        Dictionary containing the data with keys 'background', 'given_info', and 'question'
    system_prompt : str
        System prompt to be added at the beginning of the prompt

    Returns
    -------
    str
        Formatted prompt
    """
    background = datum['background'].strip()
    given_info = datum['given_info'].strip()
    question = datum['question'].strip()
    prompt = f"{system_prompt}\n{background} {given_info} {question}\nStart your answer with 'Yes' or 'No', followed by additional reasoning or evidence to support your explanation."
    return prompt

def generate_prompts(data, system_prompt):
    """
    Generate prompts for the given data

    Parameters
    ----------
    data : dict
        Dictionary containing the data with keys 'background', 'given_info', and 'question'
    system_prompt : str
        System prompt to be added at the beginning of the prompt
        
    Returns
    -------
    list
        List of formatted prompts
    """
    prompts = [format_prompt(d, system_prompt) for d in data]
    return prompts

def create_dataframe(prompts):
    """
    Create a pandas DataFrame from the list of prompts

    Parameters
    ----------
    prompts : list
        List of prompts

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the prompts
    """
    df = pd.DataFrame(prompts, columns=['prompt'])
    df = df.drop_duplicates()
    return df

def save_to_csv(df, filename):
    """
    Save the DataFrame to a CSV file
    """
    df.to_csv(filename, index=False)

def main():
    args = parse_arguments()
    data = read_data(args.input_file)
    prompts = generate_prompts(data, SYSTEM_PROMPT)
    df = create_dataframe(prompts)
    save_to_csv(df, "causal_benchmark_data.csv")

if __name__ == "__main__":
    main()