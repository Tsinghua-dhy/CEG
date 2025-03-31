import sys
sys.append(../)
from utils import ChatGPT_request
from tqdm import tqdm

topk = [ i+1 for i in range(10)]
input_file = "./datasets/case_FELM_WK_topk10.json"
def _run_nli_GPT(question, answer, passage, claim):
    prompt = "Instruction:I will show you a question, a response segment of this question, and a reference doc. Your task is to assess whether the given response segment contains factual errors or not with the help of the reference doc. If you believe the segment contains factual errors, output \"Nonfactual\"; if there is no factual error in this segment, output \"Factual\". This means that the output is \"Nonfactual\" only if there are some factual errors in the response segment. When there is no factual judgment in the response segment or the response segment has no clear meaning, you should output \"Factual\". Think it step by step. Give your reasoning first and then output the Answer.\nQuestion:\n{}\nResponse segment:\n{}\nReference doc:\n{}\nOutput:".format(question, claim, passage)
    try:
        text = ChatGPT_request(prompt)
        if "Nonfactual" in text :
            res = 0
            break 
        elif "Factual" in text :
            res = 1
            break
        else : 
            res = 0
            break
    except Exception as e:
        print(f"An error occurred: {e}")
    return res
with open(case_file, "r", encoding="utf-8") as lines:
    cases = json.load(lines)
    for topkk in topk:
        acc = 0
        acc_0 = 0
        acc_1 = 0
        num_0 = 0
        num_1 = 0
        num = 0
        idx = 0
        json_data = []
        res_file = f"./results/felm_results_topk{topk}.json"
        if os.path.exists(res_file):
            with open(res_file, "r", encoding = "utf-8" ) as json_file:
                json_data = json.load(json_file)
                idx = len(json_data)
                print(idx)
        for j, case in enumerate(tqdm(cases)):
            if j < idx :
                continue
            res=0
            premise = "\n".join([f"{i+1}.{case['ress']['res'][i]['content'].strip()}" for i in range(topkk) if case['ress']['dis'][i] > threshold])
            question = case["question"]
            res = _run_nli_GPT(question, case["answer"], premise, case["response"])
            case["res_with_GPT"] = res
            num += 1
            print(case["tag"])
            if res == case["tag"]: 
                acc += 1
            if case["tag"] == 1:
                num_1 += 1
                if res == case["tag"]:
                    acc_1 += 1
            if case["tag"] == 0:
                num_0 += 1
                if res == case["tag"]:
                    acc_0 += 1
            json_data.append(case)
            with open(res_file, "w", encoding = "utf-8" ) as json_file:
                json.dump(json_data, json_file,  ensure_ascii=False, indent=4)
        print(f"Result of file: {res_file}")
        print(f"acc = {acc/num}")
        print(f"acc_0 = {acc_0/num_0}")
        print(f"acc_1 = {acc_1/num_1}")
        print(f"balanced_acc = {(acc_0/num_0+acc_1/num_1)/2}")
