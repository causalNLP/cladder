just_scoring = True

enable_cot = False
guide_cot_until_step = [None, 1, 2, 3, 4][0]

enable_fewshot = False

model_versions = ['gpt4', 'gpt3.5long' if enable_fewshot else 'gpt3.5',
                  'gpt3.043', 'gpt3.042', 'gpt3.041', 'gpt3.04',
                  'llama007', 'alpaca007',
                  ]

openai_org_alias = 'OPENAI_ORG_ID'

enable_pdb = False

paraphrase_i = 0  # 0,1,2,3,4
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
    'arithmetic',  # TODO
][0]

openai_key_alias = 'OPENAI_API_KEY'

root_path = '../../../../2304_caubench/'
data_name = 'cladder-v1/cladder-v1-q-balanced'

models_data_name = 'cladder-v1/cladder-v1-meta-models'
with open(f'{root_path}/data/{models_data_name}.json', 'r') as f:
    import json

    meta_models = json.load(f)


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

        self.data = self.get_raw_data(file)
        self.data = self.data[:len]

    def get_ids_to_exclude(self, file=f'{root_path}/data/updated_data_ids_not_ready.txt'):
        from efficiency.log import fread
        data = '\n'.join(fread(file))
        ids = data.split(', ')
        ids = [int(i) for i in ids if i]
        return ids

    def paraphrase(self, datum):
        paraphrases = [
            "Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships",
            "Think of a self-contained, hypothetical setting with just the specified conditions, and devoid of any unknown factors or causal connections",
            "Consider a self-contained, hypothetical world with solely the mentioned conditions, and is free from any hidden factors or cause-and-effect relationships",
            "Imagine a self-contained, hypothetical setting with merely the stated conditions, and absent any unmentioned factors or causative links",
            "Think of a self-contained, hypothetical world with only the given conditions, and is void of any unknown factors or causative relationships",
        ]
        if 'given_info_para' in datum:
            datum['given_info'] = [datum['given_info'], datum['given_info_para']][paraphrase_i % 2]  # TODO: add "para"
        datum['background'] = datum['background'].replace(paraphrases[0], paraphrases[paraphrase_i])
        return datum

    def get_raw_data(self, file, inspect_data=False, exclude_query_types={}, shuffle=True):
        from efficiency.log import fread, show_var, fwrite, print_df_value_count
        from efficiency.function import random_sample, search_nested_dict

        if not shuffle: data = fread(file)
        if shuffle:
            import os
            if os.path.isfile(self.file_shuffled):
                data = fread(self.file_shuffled)
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
        data = [i for i in data if i['question_id'] not in ids_to_exclude]
        show_var(['len(data)'])

        new_data = []
        for datum in data:
            model = meta_models[datum['meta']['model_id']]
            datum['background'] = model['background']
            datum['variable_mapping'] = model['variable_mapping']
            datum['sensical'] = 1
            if model.get('nonsense', False):
                datum['sensical'] = 0
            if model.get('anticommonsense', None) is not None:
                datum['sensical'] = -1
            datum['meta']['simpson'] = model.get('simpson', False)
            datum['meta']['story_id'] = datum['meta']['story_id']
            datum['meta']['phenomenon'] = datum['meta']['graph_id']
            datum['meta']['query'] = {'query_type': datum['meta']['query_type'],
                                      'rung': datum['meta']['rung']}

            datum = self.paraphrase(datum)
            new_datum = {'old': datum}
            for k in ["question_id", "descriptive_id", ] + self.metadata_keys:
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
        'graph': 'Identify the causal graph that depicts the relationships in the scenario. {var_notions} '
                 'The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.',
        'query_type': f'Identify the type of query implied by the main question. Choices include'
                      f' {verbalize_list_of_options(query_str2id)}. '
                      f'Your answer should only be a term from the list above, enclosed in quotation marks.',
        'step1':
            'Translate the query into its formal mathematical expression based on its type, utilizing the "do(·)" notation or counterfactual notations as needed.',
        'given_info': 'Extract all the available data. Your answer should contain '
                      'nothing but marginal probabilities and conditional probabilities in the form "P(...)=..." or "P(...|...)=...", each probability being separated by a semicolon. Stick to the previously mentioned denotations for the variables.',
        'reasoning': 'Given all the information above, deduce the estimand using skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.',
        'arithmetic': 'Insert the relevant data into the estimand, perform basic arithmetic calculations, and derive the '
                      'final answer. Answer step by step.',
        'cot_final': f'Based on all the reasoning above, output one word to answer the initial question with just '
                     f'{verbalize_list_of_options(prefix2norm)}.',
    }
    q_type2step_prefix = {
        "graph": "Extract the causal graph",
        "query_type": "Determine the query type",
        "step1": "Formalize the query",
        "given_info": "Gather all relevant data",
        "reasoning": "Deduce the estimand using causal inference",
        "arithmetic": "Calculate the estimate",
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
        cot_steps = '\n\n'.join(["Guidance: Address the question by following the steps below:"] + cot_steps)
        self.q_type2prompt_suffix['cot'] = cot_steps

        for key in ['query_type', 'graph', 'step1', 'given_info', ]:
            self.q_type2prompt_suffix[key] += ' Answer concisely.'

    def __init__(self, save_path, list_of_dicts=None):
        self.init_prompt()
        self.save_path = save_path
        if list_of_dicts is not None:
            self._prepare_fewshot(list_of_dicts)
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

        var_notions = [f'Use "{s}" to represent "{t}".' for s, t in var_symb2text.items()]
        var_notions = ' '.join(var_notions)
        return var_notions

    def _prepare_fewshot(self, data):
        self.fewshot_examples = {'correlation': [20946, 22532, 14193],
                                 'ate': [2023, 28681, 27304],
                                 'marginal': [26874, 25572, 16429],
                                 'backadj': [21850, 25098, 5150],
                                 'ett': [19688, 15380, 16853],
                                 'det-counterfactual': [29884, 30531, 30821],
                                 'nde': [7775, 13272, 16351],
                                 'nie': [24877, 17209, 28444],
                                 'exp_away': [17817, 11920, 1935],
                                 'collider_bias': [6720, 7458, 2681]}
        from efficiency.function import flatten_list
        example_ids = flatten_list(self.fewshot_examples.values())
        id2datum = {i['question_id']: i for i in data if i['question_id'] in example_ids}
        self.id2fewshotprompt = {id: FewShotExamples().compose_qa_pair(datum) for id, datum in id2datum.items()}

    def _compose_fewshot_prefix(self, datum):
        exclude_id = datum['question_id']
        examples = {k: [i for i in v if i != exclude_id][0] for k, v in self.fewshot_examples.items()}
        few_shot_prompt = [self.id2fewshotprompt[i] for i in examples.values()]
        few_shot_prompt = '\n----------\n'.join(few_shot_prompt) + '\n----------\n'
        return few_shot_prompt

    def prompt_composer(self, data, enable_fewshot=enable_fewshot):
        def convert_truth_to_norm(value):
            return self.truth2norm.get(value.lower() if isinstance(value, str) else value, value)

        from copy import deepcopy

        for datum in data:
            truth_norm = convert_truth_to_norm(datum['truth'])

            q2prompt = deepcopy(self.q_type2prompt_suffix)
            q2prompt = {k: v.format(var_notions=self._datum2var_notions(datum, keep_var_values=k == 'given_info'))
                        for k, v in q2prompt.items()
                        }
            default_query_suffix = q2prompt[ask_about]
            key = 'raw_prompt_without_q' if ask_about in {'graph', 'given_info'} else 'raw_prompt'
            # TODO: add few-shot to prompt here.
            prompt = f"{datum[key]}\n\n{default_query_suffix}"
            if enable_cot:
                prompt = f"{datum[key]}\n\n{q2prompt['cot']}"
            if enable_fewshot:
                prompt = f"{self._compose_fewshot_prefix(datum)}{prompt}"
            if guide_cot_until_step:
                cot_guide = FewShotExamples().compose_qa_pair(datum, guide_cot_until_step=guide_cot_until_step)
                prompt = f"{prompt}{cot_guide}"

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


class FewShotExamples():
    from typing import Dict, Any, Iterable

    long_query_types = {
        'correlation': 'Correlation',
        'marginal': 'Marginal Distribution',
        'exp_away': 'Expaining Away Effect',
        'ate': 'Average Treatment Effect',
        'backadj': 'Backdoor Adjustment Set',
        'collider_bias': 'Collider Bias',
        'ett': 'Effect of the Treatment on the Treated',
        'nde': 'Natural Direct Effect',
        'nie': 'Natural Indirect Effect',
        'det-counterfactual': 'Counterfactual'
    }

    known_steps = ['variables', 'graph', 'query_type', 'formal_form', 'available_data',  # parsing
                   'estimand', 'estimate', 'interpretation'  # reasoning
                   ]

    def compose_qa_pair(self, datum, guide_cot_until_step=None):
        if not enable_cot:
            qa_pair = f"{datum['raw_prompt']}\n\n{datum['truth'].capitalize()}"
        else:
            if guide_cot_until_step is None:
                include_steps = self.known_steps
            else:
                include_steps = self.known_steps[:1 + guide_cot_until_step]
            step2answer = self.sub_question_prompts(datum, include_steps=include_steps)
            cot_truth = '\n\n'.join(step2answer)
            qa_pair = f"{datum['raw_prompt']}\n\n{cot_truth}"
        return qa_pair

    def sub_question_prompts(self, data: Dict[str, Any], include_steps: Iterable = ('variables', 'graph')):
        '''
        when given an unordered collection of known steps returns a dict of solved sub-questions and their answers
        when given a list or tuple of known steps returns the corresponding list of solved sub-questions and their answers

        Note: adjustement set questions are not supported
        '''
        self.q_type2step_prefix = TextInterfaceForLLMs.q_type2step_prefix
        known_steps = self.known_steps
        data = data['old']
        # data['reasoning'] = data['old']['reasoning']
        # import pdb;pdb.set_trace()

        assert all([step in known_steps for step in include_steps]), f'include_steps must be a subset of {known_steps}'

        terms = {}

        if 'variables' in include_steps:
            try:
                terms['variables'] = data['reasoning']['step0']
            except:
                # it is the "backadj" case
                return [data['answer'].capitalize()]

        if 'graph' in include_steps:
            step1 = data['reasoning']['step1'].replace(',', ', ')
            terms['graph'] = f'Step 1) {self.q_type2step_prefix["graph"]}: {step1}.'

        qtype = data['meta']['query_type']
        query_title = self.long_query_types[qtype].lower()
        if 'query_type' in include_steps:
            terms['query_type'] = f'Step 2) {self.q_type2step_prefix["query_type"]}: "{query_title}".'

        formal_form = data['meta']['formal_form']  # should be equivalent to data['reasoning']['step2']
        if 'formal_form' in include_steps:
            terms['formal_form'] = f'Step 3) {self.q_type2step_prefix["step1"]}: {formal_form}.'

        if 'available_data' in include_steps:
            step4 = data['reasoning']['step4'].replace('\n', '; ')
            if len(step4) == 0:
                # terms['available_data'] = ''
                step4 = 'No relevant data is provided'
            terms['available_data'] = f'Step 4) {self.q_type2step_prefix["given_info"]}: {step4}.'

        estimand = data['meta'].get('estimand')  # should be equivalent to data['reasoning']['step3']
        if estimand is None:  # for rung 1 questions, estimand is the same as formal_form
            estimand = formal_form
            if qtype == 'collider_bias':
                estimand = '0'
        if 'estimand' in include_steps:
            terms['estimand'] = f'Step 5) {self.q_type2step_prefix["reasoning"]}: ' \
                                f'We use causal inference to derive the estimand ' \
                                f'implied by the causal graph for the query type "{query_title}":\n' \
                                f'{formal_form}\n' \
                                f'= {estimand}'
        if 'estimate' in include_steps:
            estimate = data['reasoning']['step5']
            terms['estimate'] = f'Step 6) {self.q_type2step_prefix["arithmetic"]}:\n' \
                                f'{estimand}\n' \
                                f'= {estimate}'

        end = data['reasoning']['end']
        answer = data['answer']
        if 'interpretation' in include_steps:
            if qtype == 'collider_bias':
                reasoning = data['reasoning']['step3'].replace('.', '')
                terms['interpretation'] = f'Since {reasoning}, ' \
                                          f'the overall answer to the question is {answer}.'
            else:
                terms['interpretation'] = f'Since the estimate for the estimand is {end}, ' \
                                          f'the overall answer to the question is {answer}.'

        if isinstance(include_steps, (list, tuple)):
            return [terms[step] for step in include_steps]
        return terms


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
        df.drop(['prompt', 'question_id'], axis=1, inplace=True)

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

        write_out_files = []
        from itertools import product
        if just_scoring:
            combs = list(product(model_versions, [False], range(0, 5))) \
                    + list(product(['gpt4'], [True], range(0, 5)))
            combs = list(product(model_versions, [enable_cot], [paraphrase_i]))

        else:
            combs = list(product(model_versions, [enable_cot], [paraphrase_i]))

        print(combs)
        # if not just_scoring: import pdb;pdb.set_trace()
        if not enable_pdb:
            import pdb;
            pdb.set_trace()
        for model_version, enable_cot, para_i in combs:
            if model_version not in {'gpt4', 'gpt3.5'}:
                max_tokens += 2

            chat = Chatbot(model_version=model_version, max_tokens=max_tokens,
                           output_file=f'{root_path}/outputs/.cache_{model_version}_responses.csv',
                           system_prompt=system_prompt, openai_key_alias=openai_key_alias,
                           openai_org_alias=openai_org_alias,
                           )
            get_pred = lambda i: chat.ask(i, enable_pdb=enable_pdb)
            if enable_cot:
                get_pred = lambda i: self.cot(i, chat, max_tokens)

            cot_suffix = 'cot' if enable_cot else ''
            if cot_suffix:
                cot_suffix += str(guide_cot_until_step) if guide_cot_until_step is not None else ''
            fewshot_suffix = '_10shot' if enable_fewshot else ''
            para_suffix = f'_pa{para_i}' if para_i else ''

            write_out_file = \
                f'{root_path}/outputs/{model_version}{cot_suffix}' \
                f'{fewshot_suffix}{ask_about_suffix}{missing_step_suffix}{para_suffix}.csv'
            write_out_files.append(write_out_file)

            df_list = DataFileList()
            for data_file_obj in df_list.data_objs:
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
                        if enable_pdb:
                            print(write_out_file)
                            import pdb;
                            pdb.set_trace()
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
