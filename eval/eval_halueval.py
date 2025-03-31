import sys
sys.append(../)
from utils import ChatGPT_request, GPT_Instruct_request
import torch
import json
from tqdm import tqdm
import os
import requests
from requests.auth import HTTPBasicAuth
import sys
from retrieve import SentRetriever
from nltk.tokenize import sent_tokenize

nli_model = "instruct"

time = "0331"# give your time 

input_file = sys.argv[1]

topk = 1 # 1 or more

print("regenerate_times")
rege_times = int(sys.argv[2]) # regenerate_times: 0->1 or more

def _run_rege(case):
    url = "http://43.163.219.59:8001/beta"
    #prompt = "Premise: {}\nHypothesis: {}\nTask: Determine the logical relationship between premise and hypothesis.\nOutput format: Entailment=1, Contradiction=-1, Neutral=0.".format(passage, claim)
    prompt1 = "Instruction:I want you act as an answer judge. Given a question, two answers, your objective is to select the best and correct answer without hallucination and non-factual information. You should try your best to select the best and correct answer. If the two answers are the same, you can randomly choose one. If both answers are incorrect, choose the better one. You MUST select an answer from the provided two answers. Think it step by step. Give your reasoning first and then output your choice. Output in the following format, \"#Reasoning#:Your Reasoning\n#Choice#:Your Choice\". Your choice should be \"Answer 1\" or \"Answer 2\".\n#Question#: {}\n#Answer 1#: {}\n#Answer 2#: {}".format(case["question"], case["right_answer"], case["hallucinated_answer"])
    documents = []
    passages = []
    for seg, tag, doc in zip(case["segmented_response"], case["tags"], case["ret_knowledges"]): 
        if tag == 0:
            passages.append(seg)
            for i in range(topk):
                documents.append(doc['res'][0][k])
    prompt2 = "Documents:\n" + "\n".join(documents) + "\n"
    prompt2 += "In your previous response, there are factual inaccuracies in the following passages:" + "\n".join(f"{i}. {passage}" for i, passage in enumerate(passages, 1)) + "\nPlease re-answer the previous question with the help of documents. If the documents are unrelated to the issue, please ignore them. Output in the following format, \"#Reasoning#:Your Reasoning\n#Choice#:Your Choice\". Your choice should be \"Answer 1\" or \"Answer 2\"\n"
    #prompt2 += "In your previous response, there are factual inaccuracies in the following passages:" + "\n".join(f"{i}. {passage}" for i, passage in enumerate(passages, 1)) + "\nPlease re-answer the previous question with the help of documents. If the documents are unrelated to the issue, please ignore them. Output in the following format, \"#Reasoning#:Your Reasoning\n#Choice#:Your Choice\". Your choice should be \"Answer 1\" or \"Answer 2\". If both answers are incorrect, choose the better one. You MUST select an answer from the provided two answers." #prompt2
    print(prompt2)
    if rege_times == "0":
        prompt3 = case["text"]
    else :
        prompt3 = case["text_rege" + rege_times]
    data = [
            {"role": "user", "content": prompt1},
            {"role": "assistant", "content": prompt3},
            {"role": "user", "content":prompt2}
    ],

    try:
        text = ChatGPT_request(data)
        reasoning_start = text.find("#Reasoning#:") + len("#Reasoning#:")
        choice_start = text.find("#Choice#:")
        reasoning_text = text[reasoning_start:choice_start].strip()
        choice_text = text[choice_start + len("#Choice#:"):].strip()
        if  "Answer 1" in choice_text :
            res = 1
            break
        else :
            res = 0
            break
    except Exception as e:
        print(f"An error occurred: {e}")
    return res, text

def _run_gettag(case):
    doc = "\n".join(f"{i}. {passage[1]}" for i, passage in enumerate(case["ret_knowledge"]['res'], 1))
    prompt = "Instruction:I will show you a question, a response segment of this question, and a reference doc. Your task is to assess whether the given response segment contains factual errors or not with the help of the reference doc. If you believe the segment contains factual errors, output \"Nonfactual\"; if there is no factual error in this segment, output \"Factual\". This means that the output is \"Nonfactual\" only if there are some factual errors in the response segment. When there is no factual judgment in the response segment or the response segment has no clear meaning, you should output \"Factual\". Think it step by step. Give your reasoning first and then output the Answer.\nQuestion:\n{}\nResponse segment:\n{}\nReference doc:\n{}\nOutput:".format(case["question"], case["resp_seg"], doc)#47
    try:
        text = GPT_Instruct_request(prompt)
        reasoning_start = text.find("#Reasoning#:") + len("#Reasoning#:")
        choice_start = text.find("#Choice#:")
        reasoning_text = text[reasoning_start:choice_start].strip()
        choice_text = text[choice_start + len("#Choice#:"):].strip()
        if  "Factual" in text :
            res = 1
            break
        else :
            res = 0
            break
    except Exception as e:
        print(f"An error occurred: {e}")
    return res, text
#to segment
retrieve_data = []
with open(input_file, 'r') as file:
    input_data = json.load(file)
    for case in input_data:
        if rege_times == "0":
            text = case["text"]
        else :
            text = case["text_rege" + rege_times]
        reasoning_start = text.find("#Reasoning#:") + len("#Reasoning#:")
        choice_start = text.find("#Choice#:")
        reasoning_text = text[reasoning_start:choice_start].strip()
        reasoning_list = sent_tokenize(reasoning_text)
        case["segmented_response"] = reasoning_list
    for case in input_data:
        for seg in case["segmented_response"]:
            c = {}
            c["question"] = case["question"]
            c["right_answer"] = case["right_answer"]
            c["hallucinated_answer"] = case["hallucinated_answer"]
            c["resp_seg"] = seg
            retrieve_data.append(c)
#retrieve to get tag
ret_arrays_file = ["./results/ret_array_HaluEval_wiki3_ours_" + time + "_wiki3_GPT3" + nli_model + "_rege" + rege_times +"_" + str(ii) + ".json" for ii in range(1,3)]
topkk = 10
for idx, ret_array in enumerate(ret_arrays_file, 1):
    output_path = "psgs_w100_" + str(idx) + "_simcsebert_0130.pt"
    if os.path.exists(ret_array):
        with open(ret_array, 'r', encoding='utf-8') as json_file:
            ret_outputs = json.load(json_file)
    else :
        sentRetriever = SentRetriever("aa", output_path, model_type="simcse")
        sentRetriever.setup_faiss()
        for case in tqdm(input_data):
            for response in case["segmented_response"]:
                retriever_input = response
                ret_outputs.append(sentRetriever.faiss_retrieve(retriever_input, topkk))
        with open(ret_array, 'w', encoding='utf-8') as json_file:
            json.dump(ret_outputs, json_file, ensure_ascii=False, indent=4)
ret_arrays = []
for json_file in ret_array:
    with open(json_file, "r", encoding = "utf-8") as json_file:
        ret = json.load(json_file)
        ret_arrays.append(ret)
ret_outputs = [[] for i in range(len(retrieve_data))]
for i, ret_output in enumerate(ret_outputs):
    for ret in ret_arrays:
        ret_output.append(ret[i])
for k in range(0, len(ret_outputs)):
    all_dis = [1-d for item in ret_outputs[k] for d in item['dis']]
    topkk_indices = sorted(range(len(all_dis)), key=lambda k: all_dis[k])[:topkk]
    retrieve_data[k]["ret_knowledge"] = {}
    retrieve_data[k]["ret_knowledge"]['res'] = []
    retrieve_data[k]["ret_knowledge"]['dis'] = []
    for idx in topkk_indices:
        array_idx = idx // topkk  # 计算是哪个数组
        element_idx = idx % topkk  # 计算是数组中的哪个元素
        retrieve_data[k]["ret_knowledge"]['res'].append(ret_outputs[k][array_idx]['res'][element_idx])
        retrieve_data[k]["ret_knowledge"]['dis'].append(ret_outputs[k][array_idx]['dis'][element_idx])

#get tag
tag_file = "./results/HaluEval_response_segmented_" + time + "_wiki3_GPT3" + nli_model + "rege" + rege_times + "topk" + str(topk) + "_taged.json"
if os.path.exists(tag_file):
    with open(tag_file, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
for i, case in enumerate(tqdm(retrieve_data)):
    if i < len(json_data):
        continue
    res, text= _run_gettag(case)
    case["res"] = res
    case["text"] = text
    json_data.append(case)
    if i % 25 == 1:
        with open(tag_file, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)

#merge case
for case in input_data:
    case["ret_knowledges"] = []
    case["tags"] = []
    case["tags_text"] = []
    for segment in case["segmented_response"]:
        case["ret_knowledges"].append(retrieve_data[i]["ret_knowledge"])
        case["tags"].append(retrieve_data[i]["res"])
        case["tags_text"].append(retrieve_data[i]["text"])
        i += 1
case_file = "./datasets/case_HaluEval_ours_" + time + "_wiki3_rege" + rege_times + "_GPT3" + nli_model + "topk" + str(topk) + "_taged.json"
if os.path.exists(case_file):
    with open(case_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
else :
    cases = input_data
with open(case_file, "w", encoding="utf-8") as json_file:
    json.dump(cases, json_file, ensure_ascii=False, indent=4)

#regenerate
res_file = "./results/HaluEval_ours_" + time + "_wiki3_rege" + rege_times + "_GPT3" + nli_model + "topk" + str(topk) + "_res.json"
if os.path.exists(res_file):
    with open(res_file, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    for i, case in enumerate(tqdm(cases)):
        if i < len(json_data):
            continue
        flag = 1
        for tag in case["tags"]:
            flag = flag&tag
        if flag == 1 :
            json_data.append(case)
        else :
            res, text= _run_rege(case)
            case["res_rege" + rege_times] = res
            case["text_rege" + rege_times] = text
            json_data.append(case)
        if i % 25 == 1:
            with open(res_file, "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)
    with open(res_file, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)    