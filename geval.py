from config.azure_env import set_environment
ENVIRONMENT = set_environment()
from config.azure_keys import set_secure_keys
set_secure_keys()

from.geval_prompts import *


# load ollama client

def get_geval_score(criteria: str, steps: str, document: str, summary: str, metric_name: str):
    
    client = get_client()

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        document=document,
        summary=summary,
    )
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


evaluation_metrics = {
    "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}

def score_summaries(evaluation_metrics:dict, eval_items, summary_key:str, text_key:str = 'text'):
    from tqdm import tqdm

    summary_scores = []
    for eval_item in tqdm(eval_items):
        summary = eval_item[summary_key]
        text = eval_item[text_key]
        if summary and text:
            summary_data = {'azure_key': eval_item['azure_key'], 'summary_length': len(summary), 'text_length': len(text)}

            for metric_name, (criteria, steps) in evaluation_metrics.items():
                success = False
                for _ in range(10):
                    if not success:
                        try:
                            result = get_geval_score(criteria=criteria, 
                                                     steps = steps, 
                                                     document = text, 
                                                     summary =summary, 
                                                     metric_name=metric_name)
                            score_num = int(result.strip())
                            summary_data[metric_name] = score_num
                            success = True
                        except:
                            a=1
            summary_scores.append(summary_data)

    
    return summary_scores

def geval_on_summary_key(summary_key:str, eval_data:list, dtype:str, result_folder:str = 'enrichment/scripts/geval/results/', override = False):
    import pandas as pd
    import os
    scores = score_summaries(evaluation_metrics=evaluation_metrics,
                        eval_items = eval_data,
                        summary_key=summary_key
                        )

    df = pd.DataFrame(scores)

    fp = f'{result_folder}/{dtype}_{summary_key}.xlsx'

    if (not override) and os.path.exists(fp):
        merge_df = pd.read_excel(fp)
        keep_keys = ['azure_key', 'summary_length', 'Relevance', 'Consistency', 'Fluency', 'Coherence']
        df = df.merge(merge_df, how="outer")[keep_keys]

    df.to_excel(fp, index=False)

    return df

def get_query_dict(dtype:str):
    from enrichment.utils import map_dtype_to_sdl_source_type

    dtype_str = map_dtype_to_sdl_source_type(dtype)

    query_dict = {
        "search": "*",
        "filter": f"in_testing ne null and (sdl_source_type eq '{dtype_str}')",
        "select": "ai_summary, azure_key, abstractive_summary, text, sdl_source_type"
        }

    return query_dict


def geval(dtypes:list, current_env:str = 'dev', summary_keys = ['abstractive_summary', 'ai_summary'], max_samples = 30):
    import pandas as pd
    import os
    from search.AzureIndex import AzureIndex
    
    dtype_info = {}
    idx = AzureIndex(current_env=current_env)

    for dtype in dtypes:
        overall_info = {}

        eval_data = idx.post_query(query_dict = get_query_dict(dtype=dtype))['value']
        if len(eval_data) > max_samples:
            import random
            eval_data = random.sample(eval_data, max_samples)

        for summary_key in summary_keys:
            overall_info[summary_key] = {}
            df = geval_on_summary_key(summary_key=summary_key, 
                                      eval_data = eval_data, 
                                      dtype = dtype, 
                                      result_folder= 'enrichment/scripts/eval/results/', 
                                      override = False)

            for eval_item in ['summary_length', 'Relevance', 'Consistency', 'Fluency', 'Coherence']:
                overall_info[summary_key][eval_item] = round(float(df[eval_item].mean()), 2)

        dtype_info[dtype] = overall_info

    return dtype_info
