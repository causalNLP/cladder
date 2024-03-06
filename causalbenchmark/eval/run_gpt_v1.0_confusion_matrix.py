just_scoring = False

enable_cot = False
model_versions = ['gpt4', 'gpt3.5', 'gpt3.04', 'gpt3.043', 'gpt3.042', 'gpt3.041', ][:1]
openai_key_alias = 'OPENAI_API_KEY'

enable_pdb = False

ch_query_type = ['ett', 'ate', 'exp_away', 'cou', 'nie', 'marginal', 'collider_bias', 'nde', 'correlation'][8]
# ['ett', 'ate', 'exp_away', 'cou', 'nie', 'marginal', 'collider_bias', 'nde', 'correlation'] 0-8
# ['det-confounding', 'det-arrowhead', 'nondet-diamond', 'det-frontdoor', 'det-chain', 'frontdoor', 'chain', 'det-IV', 'det-diamondcut', 'confounding', 'mediation', 'nondet-diamondcut', 'collision', 'det-triangle', 'det-diamond', 'IV', 'det-twocauses', 'arrowhead']

# cheatsheet_type = ['query_type', 'phenomenon'][0]
# cheatsheet_id = 0

missing_step = [
    None,
    'graph',
    'query_type',
    'step1',
    'given_info',
][0]

ask_about = [
    'answer',
    'graph',
    'query_type',
    'step1',  # TODO
    'given_info',
    'reasoning',  # TODO
][0]

root_path = '../../../../2305_cladder/causalbenchmark/'
data_name = 'bern_cna_35'

class DataFileList:
    def __init__(self, file_pattern=f'{root_path}/data/{data_name}.json'):
        from glob import glob

        data_files = sorted(glob(file_pattern))
        print('Starting to get data from these files:', data_files)
        if not len(data_files):
            print('There are no files in your specified path:', file_pattern)
            import sys
            sys.exit()
        data_objs = []
        for file in data_files:
            data_obj = DataFile(file)
            data_objs.append(data_obj)
        self.data_objs = data_objs


class DataFile:
    metadata_keys = ['sensical', 'query_type', 'rung', 'phenomenon', 'simpson']

    def __init__(self, file, len=None):
        self.read_in_file = file
        self.file_shuffled = file.replace('.json', '_rand.json')
        self.file_reasoning = file.replace('.json', '_true_reasoning.json')

        self.data = self.get_raw_data(file)
        self.data = self.data[:len]


    def get_ids_to_exclude(self, file=f'{root_path}/data/updated_data_ids_not_ready.txt'):
        from efficiency.log import fread
        data = '\n'.join(fread(file))
        ids = data.split(', ')
        ids = [int(i) for i in ids if i]
        return ids

    def get_raw_data(self, file, inspect_data=False, exclude_query_types={}, shuffle=True):
        from efficiency.log import fread, show_var, fwrite, print_df_value_count
        from efficiency.function import random_sample, search_nested_dict

        if not shuffle: data = fread(file)
        if shuffle:
            import os
            if os.path.isfile(self.file_shuffled):
                data = fread(self.file_shuffled)
                # data = fread(self.file_reasoning)
                # rs = [tuple(i['reasoning'].keys()) for i in data]
                # from collections import Counter
                # Counter(rs)
                # import pdb;pdb.set_trace()
                #
                # data = {i['ID']: i for i in data}
                # data = [data[i['ID']] for i in data_ids]
            else:
                data = fread(file)
                data = random_sample(data, size=None)
                import json
                fwrite(json.dumps(data, indent=4), self.file_shuffled, verbose=True)

        datum2raw_prompt = lambda datum: f"{datum['background'].strip()}" \
                                         f" {datum['given_info'].strip()}" \
                                         f" {datum['question'].strip()}".replace(' ', ' ')
        datum2raw_prompt_without_q = lambda datum: f"{datum['background'].strip()}" \
                                                   f" {datum['given_info'].strip()}".replace(' ', ' ')
        ids_to_exclude = self.get_ids_to_exclude()
        show_var(['len(data)'])
        data = [i for i in data if i['ID'] not in ids_to_exclude]
        show_var(['len(data)'])

        new_data = []
        for datum in data:
            new_datum = {'old': datum}
            for k in ["ID", "descriptive_id", ] + self.metadata_keys:
                v = search_nested_dict(datum, k)
                if v is not None:
                    new_datum[k] = search_nested_dict(datum, k)
            new_datum.update({
                'raw_prompt': datum2raw_prompt(datum),
                'raw_prompt_without_q': datum2raw_prompt_without_q(datum),
                'truth': search_nested_dict(datum, ask_about),
            })
            # new_datum.update({k: v for k, v in datum.items() if k in {"background", "given_info", "question", }})
            new_data.append(new_datum)
        data = new_data
        if file.endswith('prop_cna_100_each.json'):
            data = [i for i in data if i['query_type'] in {'cou'}][:100]

        data = [i for i in data if i['query_type'] not in exclude_query_types]

        if inspect_data:
            import pandas as pd

            df = pd.DataFrame(data)
            columns = (set(self.metadata_keys) & df.columns) | {'truth', }
            print_df_value_count(df, columns)
            import pdb;
            pdb.set_trace()
        return data


class TextInterfaceForLLMs:
    truth2norm = {
        '1': 1,
        '0': 0,

        'yes': 1,
        'entailment': 1,
        'neutral': 0.5,
        'unknown': 0.5,
        'contradiction': 0,
        'not-counterfactual': 0,
        'counterfactual': 1,
        'no': 0,
    }

    prefix2norm = {
        'Yes': 1,
        'No': 0,
    }
    query_list_file = f'{root_path}/config/meta_queries.json'
    from efficiency.log import fread
    query_str2id = {i['query_type_str']: i['query_id'] for i in fread(query_list_file, verbose=False)}

    if ask_about == 'query_type':
        prefix2norm = query_str2id
        truth2norm = query_str2id

    from efficiency.log import verbalize_list_of_options
    q_type2prompt_suffix = {
        'answer': f'Start your answer with {verbalize_list_of_options(prefix2norm)}, followed by additional reasoning or evidence'
                  f' to support your explanation.',
        'graph': 'What is the causal graph expressed in the context? {var_notions} Answer nothing else but each edge '
                 'one by one, in the format of "var1 -> var2", and use "," to separate the edges.',
        'query_type': f'What is the query type of the above question? Choose one from'
                      f' {verbalize_list_of_options(query_str2id)}. Answer nothing else but a quoted choice from '
                      f'above.',
        'step1':
            'Based on the type of the causal query, translate the question to a formal estimand. Use the "do(·)" '
            'notation or counterfactual notations whenever necessary.',
        'given_info': 'Extract all available data. Answer nothing else but marginal probabilities and conditional '
                      'probabilities, in the form of "P(...)=..." or "P(...|...)=...", and use ";" to separate each of them. '
                      '{var_notions}',
        'reasoning': 'Given all the information above, solve for the estimand mentioned before using causal inference '
                     'skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.',
        'cot_final': f'Based on all the reasoning above, output one word to answer the initial question with just '
                     f'{verbalize_list_of_options(prefix2norm)}.',
    }
    q_type2step_prefix = {
        "graph": "Extract the causal graph",
        "query_type": "Identify the query type",
        "step1": "Translate the query to an estimand",
        "given_info": "Collect all the available data",
        "reasoning": "Solve for the estimand",
    }
    #     q_type2prompt_suffix['cot'] = f'''
    # Hint: You can answer the question by following the subquestions below:
    #
    # Step 1) Extract the causal graph: {q_type2prompt_suffix["graph"]}
    #
    # Step 2) Identify the query type: {q_type2prompt_suffix["query_type"]}
    #
    # Step 3) Translate the query to an estimand: {q_type2prompt_suffix["step1"]}
    #
    # Step 4) Collect all the available data: {q_type2prompt_suffix["given_info"]}
    #
    # Step 5) Solve for the estimand: {q_type2prompt_suffix["reasoning"]}
    #     '''.strip()

    refusal_to_answer_prefices = [
        'As a',
        "I'm sorry ",
        "neither ",
        "none ",
    ]
    prefix2norm.update({i: -1 for i in refusal_to_answer_prefices})
    prefix2norm = dict(sorted(prefix2norm.items(), key=lambda i: len(i[0]), reverse=True))

    def init_prompt(self):
        if missing_step:
            del (self.q_type2step_prefix[missing_step])
        cot_steps = [
            f'Step {i + 1}) {step}: {self.q_type2prompt_suffix[q_type]}'
            for i, (q_type, step) in enumerate(self.q_type2step_prefix.items())
        ]
        cot_steps = '\n\n'.join(["Hint: You can answer the question by following the subquestions below:"] + cot_steps)
        self.q_type2prompt_suffix['cot'] = cot_steps

        for key in ['query_type', 'graph', 'step1', 'given_info', ]:
            self.q_type2prompt_suffix[key] += ' Answer concisely.'

        ### Init cheatsheet
        file_cheatsheet = f'{root_path}/data/cheatsheets.json'

        from efficiency.log import fread
        cheatsheet = fread(file_cheatsheet)
        cheatsheet = [i for i in cheatsheet if i["sensical"] == 0]
        # c_type_instances = {i[cheatsheet_type] for i in cheatsheet}
        # len(cheatsheet)
        # from efficiency.function import set_seed, random_sample
        # set_seed()
        # cheatsheet = random_sample(cheatsheet, None)
        # c_type_instances = random_sample(c_type_instances, None)
        # c_type_this = c_type_instances[cheatsheet_id]
        self.cheatsheet = [i for i in cheatsheet if i['query_type']==ch_query_type][0]

    def __init__(self, save_path, list_of_dicts=None):
        self.init_prompt()
        self.save_path = save_path
        if list_of_dicts is not None:
            self.data_in = self.prompt_composer(list_of_dicts)

    def _datum2var_notions(self, datum, keep_var_values=False):
        var_symb_suff = 'name'
        from efficiency.function import rstrip_word
        var_symb2text = {}
        for k, v in datum['old']['variable_mapping'].items():
            if k.endswith(var_symb_suff):
                k = rstrip_word(k, var_symb_suff)
            elif keep_var_values:
                k = k[:-1] + '=' + k[-1:]
            else:
                continue
            var_symb2text[k] = v

        var_notions = [f'Use "{s}" to denote "{t}".' for s, t in var_symb2text.items()]
        var_notions = ' '.join(var_notions)
        return var_notions

    def prompt_composer(self, data):
        def convert_truth_to_norm(value):
            return self.truth2norm.get(value.lower() if isinstance(value, str) else value, value)

        from copy import deepcopy
        c = self.cheatsheet
        # c_meta = set(c.keys()) - {'cheatsheet', 'sensical'}
        data = [datum for datum in data if
                (datum['sensical'] == c['sensical']) and (datum['ID'] != c['ID'])
                # and (not any(datum[k] == c[k] for k in c_meta))
                ]

        for datum in data:
            truth_norm = convert_truth_to_norm(datum['truth'])

            q2prompt = deepcopy(self.q_type2prompt_suffix)
            q2prompt = {k: v.format(var_notions=self._datum2var_notions(datum, keep_var_values=k == 'given_info'))
                        for k, v in q2prompt.items()
                        }
            default_query_suffix = q2prompt[ask_about]
            key = 'raw_prompt_without_q' if ask_about in {'graph', 'given_info'} else 'raw_prompt'
            prompt = f"{datum[key]}\n\n{default_query_suffix}"
            if enable_cot:
                prompt = f"{datum[key]}\n\n{q2prompt['cot']}"
            prompt = f"First read an example question with its step-by-step answer, and then answer the next " \
                     f"question.\n\n{c['cheatsheet']}\n\n{prompt}"

            del datum['raw_prompt'], datum['raw_prompt_without_q'], datum['old']
            datum.update({
                'prompt': prompt,
                'truth_norm': truth_norm,
            })

        return data

    def response_processor(self, **kwargs):
        def convert_to_norm(value):
            invalid = -1
            value = str(value).lower().strip().strip('"')

            for prefix, norm in self.prefix2norm.items():
                if value.startswith(prefix.lower()):
                    return norm
            return invalid

        from efficiency.log import fread
        data = fread(self.save_path)
        if data:
            self.data_out = data

            for datum in self.data_out:
                datum['pred_norm'] = convert_to_norm(datum['pred'])
                datum.update(kwargs)
            self.save()

    def save(self):
        from efficiency.log import write_dict_to_csv
        write_dict_to_csv(self.data_out, self.save_path, verbose=True)
        # import pandas as pd
        # df = pd.DataFrame(self.data)
        # df.to_csv(self.save(), index=False)
        # print('')


class Scorer:
    def __init__(self, files, data_list_of_dicts=None):
        if not len(files):
            print('No files for evaluation')
            import sys
            sys.exit()

        from efficiency.log import fread
        data_list = []
        for file in sorted(files):
            data = fread(file)
            data_list += data
            print(file, len(data))

        import pandas as pd
        df = pd.DataFrame(data_list)
        # df = pd.read_csv(file, index_col=None)
        self.truth_pred_scorer(df)

    def apply_score_func(self, df, pred_key='pred_norm', truth_key='truth_norm'):
        if ask_about in {'graph'}:
            pred_key = 'pred'
            truth_key = 'truth'

            def score_func(row):
                def txt2edges(txt):
                    txt = txt.replace(' ', '')
                    edges = txt.split(',')
                    edges = {tuple(sorted(i.split('->', 1))) for i in edges}
                    return edges

                def edge_set2node_set(edges):
                    from efficiency.function import flatten_list
                    nodes = flatten_list(edges)
                    return set(nodes)

                from efficiency.function import get_set_f1, get_set_edit_distance

                pred_edges = txt2edges(row[pred_key])
                truth_edges = txt2edges(row[truth_key])
                edge_f1 = get_set_f1(truth_edges, pred_edges)

                pred_nodes = edge_set2node_set(pred_edges)
                truth_nodes = edge_set2node_set(truth_edges)
                node_f1 = get_set_f1(truth_nodes, pred_nodes)

                edit_distance = get_set_edit_distance(truth_edges, pred_edges)
                score_dict = {
                    'node_f1': node_f1,
                    'edge_f1': edge_f1,
                    'edge_edit_distance': edit_distance,
                    'score': edge_f1,
                }
                return score_dict
        else:
            # if ask_about in {'answer', 'query_type'}:
            score_func = lambda row: {'score': row[pred_key] == row[truth_key]}

        # df['score'] = df.apply(score_func, axis=1)
        import pandas as pd
        score_df = df.apply(lambda row: pd.Series(score_func(row)), axis=1)
        df = df.join(score_df)
        print(score_df.mean())
        score_df.describe()

        import pdb;
        pdb.set_trace()
        return df

    def truth_pred_scorer(self, df):
        df.drop(['prompt', 'ID', 'descriptive_id'], axis=1, inplace=True)

        df = self.apply_score_func(df)
        # df['score'] = (df['pred_norm'] == df['truth_norm'])
        import pdb;
        pdb.set_trace()

        if ask_about not in {'graph'}:
            from sklearn.metrics import classification_report
            df_valid = df[~df['pred_norm'].isna()]
            for rung in [1, 2, 3]:
                report = classification_report(df_valid[df_valid['rung'] == rung]['truth_norm'],
                                               df_valid[df_valid['rung'] == rung]['pred_norm'], digits=4)
                print(report)
            report = classification_report(df_valid['truth_norm'], df_valid['pred_norm'], digits=4)
            print(report)

        import pdb;
        pdb.set_trace()

        res_dfs = []
        for uniq_vign_key in ['model_version']:
            try:
                res_df = self._res_by_group(df, uniq_vign_key)
                res_dfs.append(res_df)
            except:
                continue
        for model_version in sorted(df['model_version'].unique().tolist()):
            new_df = df[df['model_version'] == model_version]
            for uniq_vign_key in DataFile.metadata_keys:
                try:
                    res_df = self._res_by_group(new_df, uniq_vign_key)
                    res_df['model_version'] = model_version
                    res_dfs.append(res_df)
                except:
                    continue
            import pdb;
            pdb.set_trace()
        import pandas as pd
        res_df = pd.concat(res_dfs)
        print(res_df)
        res_df.to_csv(f'{root_path}/outputs/performance.csv')

        import pdb;
        pdb.set_trace()
        from efficiency.log import pivot_df
        pivot_df(res_df)
        import pdb;
        pdb.set_trace()

    @staticmethod
    def _res_by_group(df, uniq_vign_key, result_key='score', return_obj=['group_dict', 'consistency_rate'][0]):
        # Group by 'group' column and count the occurrences of each value in the 'result' column
        g = df.groupby(uniq_vign_key)[result_key]
        dff = round(g.mean() * 100, 2).reset_index()
        dff['count'] = g.count().to_list()
        print(dff)
        return dff

        g_counts = df.groupby(uniq_vign_key)[result_key].value_counts()
        g_counts.name = 'performance'  # otherwise, there will be an error saying that `result_key` is used
        # for both the name of the pd.Series object, and a column name
        g_totals = g_counts.groupby(uniq_vign_key).sum()
        g_perc = round(g_counts / g_totals * 100, 2)
        g_major = g_perc.groupby(uniq_vign_key).max()
        consistency_rate = round(g_major.mean(), 2)

        if return_obj == 'group_dict':
            g_perc_clean = g_perc.drop([False],
                                       level=result_key, errors='ignore')
            # dff = g_perc_clean.reset_index() # turn into df
            # g_perc_clean.to_csv(performance_file)

            print(g_perc_clean)
            # print('[Info] The above results are saved to', performance_file)

            return g_perc_clean.to_dict()
        elif return_obj == 'consistency_rate':
            return consistency_rate


class Tester:
    def __init__(self):
        from efficiency.function import set_seed
        set_seed()

    def cot(self, query, chat, max_tokens):

        datum = {}
        queries = [query, TextInterfaceForLLMs.q_type2prompt_suffix['cot_final']]
        for query_i, query in enumerate(queries):
            response = chat.ask(
                query, continued_questions=query_i,
                turn_off_cache=query_i,
                max_tokens=max_tokens if query_i else 1024,
                enable_pdb=enable_pdb,
            )

            datum[f'query{query_i}'] = query
            datum[f'pred{query_i}'] = response
        return response

    def run_default_test(self, just_scoring=just_scoring, enable_cot=enable_cot):
        from efficiency.nlp import Chatbot
        system_prompt = '''
You are an expert in causal inference. The following question is not a typical commonsense query, but rather a meticulously designed question created by a professor specializing in causal inference, intended to assess the students' mastery of the course content.
        '''.strip()
        max_tokens = 200
        if ask_about == 'answer':
            max_tokens = 1
        elif ask_about == 'query_type':
            max_tokens = 20

        from tqdm import tqdm
        import pandas as pd
        ask_about_suffix = f'_{ask_about.replace("_", "-")}' if ask_about != 'answer' else ''
        missing_step_suffix = f'_no{missing_step.replace("_", "-")}' if missing_step else ''
        cheat_suffix = f'_ch-{ch_query_type.replace("_", "-")}' if ch_query_type else ''

        write_out_files = []
        from itertools import product
        if just_scoring:
            combs = list(product(model_versions, [False])) + [('gpt4', True)]
        else:
            combs = list(product(model_versions, [enable_cot]))

        print(combs)
        if not just_scoring: import pdb;pdb.set_trace()
        for model_version, enable_cot in combs:
            if model_version not in {'gpt4', 'gpt3.5'}:
                max_tokens += 10
            chat = Chatbot(model_version=model_version, max_tokens=max_tokens,
                           output_file=f'{root_path}/outputs/.cache_{model_version}_responses.csv',
                           system_prompt=system_prompt, openai_key_alias=openai_key_alias,
                           )
            get_pred = lambda i: chat.ask(i, enable_pdb=enable_pdb)
            if enable_cot:
                get_pred = lambda i: self.cot(i, chat, max_tokens)

            cot_suffix = 'cot' if enable_cot else ''
            write_out_file = \
                f'{root_path}/outputs/{data_name}_{model_version}{cot_suffix}{ask_about_suffix}{missing_step_suffix}{cheat_suffix}.csv'
            write_out_files.append(write_out_file)

            for data_file_obj in DataFileList().data_objs:
                if not just_scoring:
                    data_obj = TextInterfaceForLLMs(write_out_file, data_file_obj.data, )
                    data = data_obj.data_in

                    tqdm_desc = f'Model={chat.model_version}, Data={write_out_file}'

                    print(tqdm_desc)
                    for datum_i, datum in tqdm(list(enumerate(data)), desc=tqdm_desc):
                        query = datum['prompt']
                        pred = get_pred(query)
                        datum['pred'] = pred

                        df = pd.DataFrame(data[:datum_i + 1])
                        df.to_csv(write_out_file, index=False)
                else:
                    data_obj = TextInterfaceForLLMs(write_out_file)

                data_obj.response_processor(model_version=f"{model_version}{cot_suffix}")

        scorer = Scorer(write_out_files)


def main():
    tester = Tester()
    tester.run_default_test()


if __name__ == '__main__':
    main()
