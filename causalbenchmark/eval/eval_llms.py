import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import json
import argparse
def getOriginalDataset(originalFile):
    with open(originalFile) as json_file:
        data = json.load(json_file)
    system_prompt = '''
    You are an expert in causal inference. The following question is not a typical commonsense query, but rather a meticulously designed question created by a professor specializing in causal inference, intended to assess the students' mastery of the course content.
    '''
    system_prompt=system_prompt.strip()
    datum2raw_prompt = lambda datum: f"{datum['background'].strip()}" \
                                             f" {datum['given_info'].strip()}" \
                                             f" {datum['question'].strip()}".replace(' ', ' ')
    prompts=[]
    sensical=[]
    rung=[]
    ans=[]
    ids=[]
    query_type=[]
    for d in data:
        prompt=datum2raw_prompt(d)+'\nStart your answer with "Yes" or "No", followed by additional reasoning or evidence to support your explanation.'
        prompts.append(prompt)
        ids.append(d['ID'])
        try:
            rung.append(d['meta']['query']['rung'])
        except:
            rung.append('0')
        sensical.append(d['sensical'])
        ans.append(d['answer'])
        query_type.append(d['meta']['query']['query_type'])

    df_org=pd.DataFrame({'query':prompts,'sensical':sensical,'rung':rung,'ans':ans,'ID':ids,'query_type':query_type})
    return df_org

def removeIDSNotReady(df_org):
    dataNotReady="../../data/updated_data_ids_not_ready.txt"
    #dataNotReady="./updated_data_ids_not_ready.txt"
    with open(dataNotReady, "r") as file:
        lines = file.readlines()
    ids_not_ready=lines[0].split(', ')
    ids_not_ready=[int(i) for i in ids_not_ready]
    df_org=df_org.loc[~df_org.ID.isin(ids_not_ready),:]
    return df_org

def main(args,originalFile,queryTypeFile):

    predictionFile=args['predFile']
    scoresFile=args['outputFile']
    ## dictionary to infer rung number
    df_lookup = pd.read_csv(queryTypeFile)
    query_type2rung = dict(zip(df_lookup['alias'], df_lookup['layer1']))

    predDF=pd.read_csv(predictionFile)
    df_org=getOriginalDataset(originalFile)

    df_org['rung'] = df_org['query_type'].map(query_type2rung)
    df_org=removeIDSNotReady(df_org)
    predDF.pred=predDF.pred.str.strip()
    predDF=predDF.assign(pred=np.where(predDF.pred.str.lower().str.startswith('yes'),'yes',
                                      np.where(predDF.pred.str.lower().str.startswith('no'),'no',None)))
    predDF=predDF.loc[:,['pred','query']]
    df_org=df_org.merge(predDF,on='query')
    
    scores={}
    scores['accuracy']=accuracy_score(df_org.ans,df_org.pred)

    for r in ['Rung 1', 'Rung 2', 'Rung 3']:
        df_temp=df_org.copy()
        df_temp=df_temp.loc[df_temp.rung==r]
        scores['accuracy_'+r]=accuracy_score(df_temp.ans,df_temp['pred'])

    for r in [1,-1,0]:
        df_temp=df_org.copy()
        df_temp=df_temp.loc[df_temp.sensical==r]
        scores['accuracy_sensical_'+str(r)]=accuracy_score(df_temp.ans,df_temp['pred'])

    with open(scoresFile, 'w') as convert_file:
         convert_file.write(json.dumps(scores))
    
    print(scores)

if __name__ == '__main__':
# parse from arguments the dataset to be used
    args=argparse.ArgumentParser()
    args.add_argument("--predFile",type=str,default="./.cache_llama007_responses.csv")
    args.add_argument("--outputFile",type=str,default="./scores.txt")
    args=vars(args.parse_args())
    originalFile="../../data/bern_cna_35.json"
    queryTypeFile = '../../data/data_stats.csv'
    main(args,originalFile,queryTypeFile)