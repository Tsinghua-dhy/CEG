import sys
sys.append(../)
from utils import GPT_Instruct_request
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def _run_nli_GPT(passage, claim):
    prompt = "Context:\n{}\nSentence:\n{}\nIs the sentence supported by the context above? Answer Yes or 
    try:
        text = GPT_Instruct_request(prompt)
        if  "Yes" in text or "yes" in text:
            res = 1
        else :
            res = 0
    except Exception as e:
        print(f"An error occurred: {e}")
    return res, text
case_file = "./datasets/case_wikibio.json"
topkk = [ i+1 for i in range(10)]
with open(case_file, "r", encoding="utf-8") as lines:
    case_segs = json.load(lines)
    acc = 0
    num = 0
    i = 0
    for topk in topkk:
        res_file = f"./results/wikibio_result_topk{topk}.json"
        if os.path.exists(res_file):
            with open(res_file, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)
        for j, case_seg in enumerate(tqdm(case_segs)):
            for case in tqdm(case_seg):
                i += 1
                if i <= len(json_data):
                    continue
                res = 0
                premise= "\n".join([f"{i+1}.{case['res']['res'][i]['content']}" for i in range(topk)])
                res, text = _run_nli_GPT(premise, case["sentence"])
                case["res_with_GPT3"] = res
                case["response_of_GPT3"] = text
                num += 1
                if res == case["tag"]:
                    acc += 1 
                json_data.append(case)
                if i%25 == 1:
                    with open(res_file, "w", encoding="utf-8") as json_file:
                        json.dump(json_data, json_file, ensure_ascii=False, indent=4)
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)
