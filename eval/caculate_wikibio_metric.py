from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import json
import sys

# 初始化列表用于存储真实标签和预测分数
y_true = []
y_scores = []
acc = 0
num = 0
num_0 = 0
num_1 = 0
acc_0 = 0
acc_1 = 0
# 从文件中读取数据    
topkk = [i+1 for i in range(10)]
for topk in topkk:
    res_file = f"./results/wikibio_result_topk{topk}.json"
    with open(input_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        for case in cases:
            if case["tag"] == 0.5:
                case["tag"] = 0
            num += 1
            pred = case["pred_of_GPT3"]   
            if case["tag"] == 0:
                if pred == case["tag"]:
                    acc_0 += 1
                    acc += 1
                num_0 += 1
            else:
                if pred == case["tag"]:
                    acc_1 += 1
                    acc += 1
                num_1 += 1
            y_true.append(case["tag"])
            y_scores.append(pred)
            
    y_true = np.eye(2)[y_true]
    y_scores = np.eye(2)[y_scores]

    n_classes = 2

    # 初始化字典用于存储每个类别的Precision和Recall
    precision = dict()
    recall = dict()
    pr_auc = dict()

    # 计算每个类别的Precision和Recall
    for i in range(n_classes):
        # 输出新的列表
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # 输出每个类别的PR-AUC
    print(f"Results of file: {res_file}")
    for i in range(n_classes):
        print(f'Class {i} PR-AUC: {pr_auc[i]*100}')
    print(f'acc: {acc/num*100}')
    print(f'acc_0: {acc_0/num_0*100}')
    print(f'acc_1: {acc_1/num_1*100}')
    print(f'balanced acc: {(acc_0/num_0 + acc_1/num_1)/2*100}')

    # Output the values again
    for i in range(n_classes):
        print(f'{pr_auc[i]*100:.2f}', end=' ')
    print(f'{acc_0/num_0*100:.2f}', end=' ')
    print(f'{acc_1/num_1*100:.2f}', end=' ')
    print(f'{(acc_0/num_0 + acc_1/num_1)/2*100:.2f}')
