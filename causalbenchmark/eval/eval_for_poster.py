import os
import openai
import itertools
import json
from tqdm import tqdm

open_api_key = os.environ['OPENAI_API_KEY']



# Super naive scorer
def tf_scorer(pred, truth):
    false_family = ['no', 'false', 'incorrect', 'not necessarily']
    true_family = ['yes','true','correct']

    pred_in_false = any(pred in element for element in false_family)
    truth_in_false = any(truth in element for element in false_family)
    pred_in_true = any(pred in element for element in true_family)
    truth_in_true = any(truth in element for element in true_family)

    # if type(truth) == int:
    #     if pred_in_false and not truth:
    #         return 1
    #     elif pred_in_true and truth:
    #         return 1
    #     return 0
    #
    #
    if pred_in_false and truth_in_false:
        return 1
    elif pred_in_true and truth_in_true:
        return 1
    else:
        return 0

def eval_direct_dataset(file_path,eval_function):
    to_save = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        for row in tqdm(data, desc="Processing rows"):
            prompt = row['background'] + row['question']
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=100,
                messages=[
                    {"role": "user", "content": prompt}
                ])
            response = completion.choices[0].message["content"]

            prediction = eval_function(response, row['answer'])
        conv = {"Prompt": prompt,
                'response': response,
                'ground_truth': row['answer'],
                'prediction_correct': prediction}

        to_save.append(conv)

        # Save the conversation data to a new JSON file
        input_file_name = os.path.basename(file_path)
        input_file_base, input_file_ext = os.path.splitext(input_file_name)
        new_file_name = f"{input_file_base}_llm_response{input_file_ext}"
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
    print('running')

    with open(new_file_path,'w') as f:
        json.dump(to_save, f, indent=4)

    return new_file_path


direct_det_path = eval_direct_dataset('../../data/data_sampled.json', tf_scorer)
print(f"The new file is saved at: {direct_det_path}")
direct_nondet_path = eval_direct_dataset('../../data/nondet_sampled.json', tf_scorer)
print(f"The new file is saved at: {direct_nondet_path}")


def eval_graph_dataset(file_path,eval_function):
    to_save = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        for row in tqdm(data, desc="Processing rows"):
            prompt = row['background'] + row['question'] + \
                     "Answer the question by first drawing the causal graph for the story"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=100,
                messages=[
                    {"role": "user", "content": prompt}
                ])
            response = completion.choices[0].message["content"]

            prediction = eval_function(response, row['answer'])
        conv = {"Prompt": prompt,
                'response': response,
                'ground_truth': row['answer'],
                'prediction_correct': prediction}

        to_save.append(conv)

        # Save the conversation data to a new JSON file
        input_file_name = os.path.basename(file_path)
        input_file_base, input_file_ext = os.path.splitext(input_file_name)
        new_file_name = f"{input_file_base}_llm_response{input_file_ext}"
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    with open(new_file_path,'w') as f:
        json.dump(to_save, f, indent=4)

    return new_file_path


graph_det_path = eval_graph_dataset('../../data/data_sampled.json', tf_scorer)
print(f"The new file is saved at: {graph_det_path}")
graph_nondet_path = eval_graph_dataset('../../data/nondet_sampled.json', tf_scorer)
print(f"The new file is saved at: {graph_nondet_path}")

def eval_cs_dataset(file_path,eval_function):
    to_save = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        for row in tqdm(data, desc="Processing rows"):
            prompt = row['cheatsheet'] + "\n" + row['background'] + row['question']
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=100,
                messages=[
                    {"role": "user", "content": prompt}
                ])
            response = completion.choices[0].message["content"]

            prediction = eval_function(response, row['answer'])
        conv = {"Prompt": prompt,
                'response': response,
                'ground_truth': row['answer'],
                'prediction_correct': prediction}

        to_save.append(conv)

        # Save the conversation data to a new JSON file
        input_file_name = os.path.basename(file_path)
        input_file_base, input_file_ext = os.path.splitext(input_file_name)
        new_file_name = f"{input_file_base}_llm_response{input_file_ext}"
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
    print('running')

    with open(new_file_path,'w') as f:
        json.dump(to_save, f, indent=4)

    return new_file_path


cs_det_path = eval_cs_dataset('../../data/data_sampled_cs.json', tf_scorer)
print(f"The new file is saved at: {cs_det_path}")

cs_nondet_path = eval_cs_dataset('../../data/nondet_sampled_cs.json', tf_scorer)
print(f"The new file is saved at: {cs_nondet_path}")





def eval_subq_dataset(file_path,eval_function):
    to_save = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        for row in tqdm(data, desc="Processing rows"):
            prompt = row['background'] + row['subquestion1']
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=100,
                messages=[
                    {"role": "user", "content": prompt}
                ])
            response1 = completion.choices[0].message["content"]
            conversation_id = completion["conversation_id"]
            followup = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=100,
                conversation_id = conversation_id,
                messages=[
                    {"role": "user", "content": prompt}
                ])

            response2 = followup.choices[0].message["content"]

            prediction = eval_function(response2, row['answer'])
        conv = {"Prompt": prompt,
                "Subquestion1": row['subquestion1'],
                'intermediate-response': response1,
                "query": row['question'],
                'response': response2,
                'ground_truth': row['answer'],
                'prediction_correct': prediction}

        to_save.append(conv)

        # Save the conversation data to a new JSON file
        input_file_name = os.path.basename(file_path)
        input_file_base, input_file_ext = os.path.splitext(input_file_name)
        new_file_name = f"{input_file_base}_llm_response{input_file_ext}"
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
    print('running')

    with open(new_file_path,'w') as f:
        json.dump(to_save, f, indent=4)

    return new_file_path


subq_det_path = eval_subq_dataset('../../data/data_sampled_cs.json', tf_scorer)
print(f"The new file is saved at: {subq_det_path}")

subq_nondet_path = eval_subq_dataset('../../data/nondet_sampled_cs.json', tf_scorer)
print(f"The new file is saved at: {subq_nondet_path}")





