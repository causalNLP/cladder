file = '../outputs/given-info_human_eval.csv'
from efficiency.log import fread
from efficiency.function import get_set_f1

df = fread(file, return_df=True)

# Forward-fill missing values in column 'A'
df['ID'] = df['ID'].fillna(method='ffill')
df['gpt_prediction_about_this'] = df.apply(
    lambda row: row['ground_truth_given_info']
    if row['this_term_is_mentioned'] == 1
       and row['the_quantity_is_correct'] == 1
    else row['gpt_prediction_about_this'],
    axis=1)
print(df)
df.to_csv('../outputs/given-info_human_eval_reformat.csv', index=False)

grouped = df.groupby('ID')['ground_truth_given_info'].apply(list)
id2truth = grouped.to_dict()

grouped = df.groupby('ID')['gpt_prediction_about_this'].apply(list)
id2pred = grouped.to_dict()

f1s = []
for row_id, truth_set in id2truth.items():
    pred_set = id2pred[row_id]
    f1 = get_set_f1(truth_set, pred_set)
    f1s.append(f1)
from efficiency.function import avg
import pdb; pdb.set_trace()
import numpy as np
avg(f1s)
np.std(f1s)