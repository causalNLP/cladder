import os
import itertools
import json
import random
from pathlib import Path
from omnibelt import export
import omnifig as fig
from causalbenchmark import util


def sample_data_points(file_path, num):
    # Load the data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Group data points by phenomenon
    grouped_data = {}
    for data_point in data:
        phenomenon = data_point['meta']['phenomenon']

        if phenomenon not in grouped_data:
            grouped_data[phenomenon] = []

        grouped_data[phenomenon].append(data_point)

    # Randomly sample num data points for each phenomenon
    sampled_data = []
    for key, value in grouped_data.items():
        if len(value) >= num:
            sampled_data += random.sample(value, num)
        else:
            print(f"Not enough data points for {key}. Only {len(value)} available. Taking all available data points.")
            sampled_data += value

    # Save the sampled data to a new JSON file
    input_file_name = os.path.basename(file_path)
    input_file_base, input_file_ext = os.path.splitext(input_file_name)
    new_file_name = f"{input_file_base}_sampled{input_file_ext}"
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    with open(new_file_path, 'w') as file:
        json.dump(sampled_data, file, indent=4)

    return new_file_path


# Call the function with the path to your data file and the desired number of samples


nondet_new_file_path = sample_data_points('data/nondet.json', 100)
print(f"The new file is saved at: {nondet_new_file_path}")



# create a files with sub-questions
def add_subs(file_path):
    # Load the data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    data_subq = data.copy()
    for row in data_subq:
        story = util.load_story_from_config(row['meta']['story_id'])
        if "Nonident" in row['meta']['query']:
            row['subquestion1'] = f"What values are necessary in order to compute the causal effect of {story['labels']['Xname']} on {story['labels']['Yname']}?"
        if row['meta']['phenomenon'] == 'confoinding':
            row['subquestion1'] = f"What is the confounder between {story['labels']['Xname']} and {story['labels']['Yname']}?"
        elif row['meta']['phenomenon'] == 'chain':
            row['subquestion1'] = f"Is it {story['labels']['Xname']} or {story['labels']['Zname']} that itself is the cause of {story['labels']['Yname']}?"
        elif row['meta']['phenomenon'] == 'mediator' or row['meta']['phenomenon'] == 'arrowheadmediation':
            if 'nie' in row['meta']['query']:
                row['subquestion1'] = f"To access the effect of {story['labels']['Xname']} on {story['labels']['Yname']} " \
                                      f"through affecting other variables, what variable should be controlled for?"
            else:
                row['subquestion1'] = f"To access the direct effect of {story['labels']['Xname']} on {story['labels']['Yname']}, " \
                                      f"what variable should be controlled for?"

        elif row['meta']['query'] == "PropCounterfactualEffectQuery":
            row['subquestion1'] = f"What would happen to {story['labels']['Yname']}"
        elif row['meta']['query'] == "PropCounterfactualEffectQuery":
            continue
        elif row['meta']['phenomenon'] == 'collision' or row['meta']['phenomenon'] == 'arrowheadcollision':
            row['subquestion1'] = f"Explain the correlation between {story['labels']['Xname']} and {story['labels']['Yname']} " \
                               f"from a causal inference perspective."
        elif row['meta']['phenomenon'] == 'IV':
            row['subquestion1'] = f"What is an instrumental variable for the effect of {story['labels']['Xname']} on {story['labels']['Yname']}"
        elif row['meta']['phenomenon'] == 'frontdoor':
            row['subquestion1'] = f"What is a valid adjustment set of {story['labels']['Xname']} and {story['labels']['Yname']}?"

        # Save the sampled data to a new JSON file
        input_file_name = os.path.basename(file_path)
        input_file_base, input_file_ext = os.path.splitext(input_file_name)
        new_file_name = f"{input_file_base}_subq{input_file_ext}"
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    with open(new_file_path, 'w') as file:
        json.dump(data_subq, file, indent=4)

    return new_file_path




def add_cheatsheet(file_path):
    # Load the data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    data_cs = data.copy()
    for row in data_cs:
        if "Prop" in row['meta']['query']:
            query = util.load_query_from_config('cou_y')
        elif row['meta']['query'] == 'AverageTreatmentEffectQuery' or row['meta']['query'] == 'NonidentATEQuery':
            query = util.load_query_from_config('ate')
        elif row['meta']['query'] == 'EffectTreatmentTreatedQuery' or row['meta']['query'] == 'NonidentETTQuery':
            query = util.load_query_from_config('ett')
        else:
            print(row['meta']['query'])
        row['cheatsheet'] = query['cheatsheet']


    # Save the data to a new JSON file
    input_file_name = os.path.basename(file_path)
    input_file_base, input_file_ext = os.path.splitext(input_file_name)
    new_file_name = f"{input_file_base}_cs{input_file_ext}"
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    with open(new_file_path, 'w') as file:
        json.dump(data_cs, file, indent=4)

    return new_file_path


det_subq_file_path = add_subs(det_new_file_path)
print(f"The new file is saved at: {det_subq_file_path}")

nondet_subq_file_path = add_subs(nondet_new_file_path)
print(f"The new file is saved at: {nondet_subq_file_path}")

det_cheatsheet_file_path = add_cheatsheet(det_new_file_path)
print(f"The new file is saved at: {det_cheatsheet_file_path}")

nondet_cheatsheet_file_path = add_cheatsheet(nondet_new_file_path)
print(f"The new file is saved at: {nondet_cheatsheet_file_path}")











