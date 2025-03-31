import json

input_files = # results of different regenerate times
for input_file in input_files:
    with open(input_file, "r", encoding="utf-8") as lines:
        data = json.load(lines)
        num = 0
        acc =0
        for case in data:
            res = case["res"]
            if res == 1:
                acc += 1
            num += 1
        print(acc/num)