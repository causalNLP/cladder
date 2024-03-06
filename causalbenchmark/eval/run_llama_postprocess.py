def main():
    import os

    folder='../../outputs/'

    for model_name in ['llama007', 'alpaca007']:
        llm_file = folder + f'.cache_{model_name}_responses.csv'
        if not os.path.isfile(llm_file):
            continue

        cache = load_cache(llm_file)
        data_file = folder + 'gpt3.5.csv'
        data_file_new = folder + f'{model_name}.csv'
        from efficiency.log import fread
        data = fread(data_file)
        for datum in data:
            prompt_llm = datum['prompt'].split('\n', 1)[0].strip()
            if prompt_llm not in cache:
                import pdb;pdb.set_trace()
            datum['pred'] = cache[prompt_llm]
            datum['pred_norm'] = convert_to_norm(datum['pred'])
        from efficiency.log import write_dict_to_csv
        write_dict_to_csv(data, data_file_new, verbose=True)
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
def load_cache(file):
    cache = {}
    from efficiency.log import fread
    data = fread(file, verbose=False)
    cache.update({i[f'query{q_i}']: i[f'pred{q_i}'] for i in data
                  for q_i in list(range(10)) + ['']
                  if f'query{q_i}' in i})
    cache = {k.replace('\n', ' '): v for k, v in cache.items() if v}  # there are cases where the response is empty

    return cache
if __name__ == '__main__':
    main()