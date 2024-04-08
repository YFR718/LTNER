# 实现具体的类
from common.chatgpt.gpt import getGpt, GPT35T, ADA3Small
from common.vector import zilliz

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import Any, List, Union
from tqdm import tqdm


import re

from data.loader.mapLoader import load


class MyNERExperience():
    def __init__(self, data_path: str, save_path: str,fewnum):
        self.dataPath = data_path  # 数据路径
        self.savePath = save_path  # 结果保存路径
        self.TP = 0  # 正确的正例，即模型正确预测为正类的样本数量
        self.FP = 0  # 错误的正例，即模型错误预测为正类的样本数量（实际上是负类）
        self.TN = 0  # 正确的负例，即模型正确预测为负类的样本数量
        self.FN = 0  # 错误的负例，即模型错误预测为负类的样本数量（实际上是正类）
        self.fewNum = fewnum


    def preprocessing(self) -> List:
        # 具体的预处理实现
        return []

    def dataloader(self) -> List:
        # 具体的数据加载实现
        return load(self.dataPath)

    def getPrompts(self, input: Any) -> Any:
        return ""
    def process_strings(string_list):
        return [s.strip().lower() for s in string_list]

    def addResults(self, predict: Any, truth: Any):
        for k in ["PER","LOC","ORG","MISC"]:
            if not predict.__contains__(k):
                predict[k] = []
        for k in predict.keys():
            predict[k] = [str(s).strip().lower() for s in predict[k]]
        for k in truth.keys():
            truth[k] = [str(s).strip().lower() for s in truth[k]]
        # print( "truth:",truth,"predit:",predict)
        # 遍历所有预测的键和值
        for key in predict.keys():
            # 每个类别的预测实体列表
            pred_entities = predict[key]
            # 每个类别的真实实体列表，若不存在则为空列表
            true_entities = truth.get(key, [])

            # 计算TP和FP
            for entity in pred_entities:
                if entity in true_entities:
                    # print(key," TP:",entity)
                    self.TP += 1  # 真实实体列表中含有此预测实体，记为TP
                else:
                    # print(key," FP:", entity)
                    self.FP += 1  # 不在真实实体列表中，记为FP

            # 计算FN
            for entity in true_entities:
                if entity not in pred_entities:
                    # print(key," FN:", entity)
                    self.FN += 1  # 真实实体未被预测出，记为FN

    def run(self):
        data = self.dataloader()
        records = []
        # i = 0
        # 创建一个最大并发度为10的ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=50) as executor:
            # 使用executor.map并行处理数据
            futures = [executor.submit(self.process_data, d) for d in data[:]]
            # 使用tqdm创建进度条
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing'):
                # i+=1
                # print(i)
                record = future.result()
                records.append(record)
        P, R, F1 = self.getPRF1()
        records.append({"P": P, "R": R, "F1": F1, "TP": self.TP, "FN": self.FN, "FP": self.FP})
        print(records[-1])
        self.record(records)

    def process_data(self, d):

        s = ""
        retries = 0
        while retries < 5:
            try:
                fews = zilliz.searchSentence(d["sentence"], "conll003baseline", ADA3Small, self.fewNum,["ner", "sentence"])

                prompt = [{
                    "role": "assistant", "content":f"""You are an excelent linquist. The task is to label ORG、PER、LOC、MISC  entities  in the given sentence. use ## and ##+Type to to mark targets."""
                }]

                for f in fews:
                    prompt.append({
                        "role": "assistant",
                        "content": f['entity']['sentence']})
                    content = f['entity']['sentence']
                    m = json.loads(f['entity']['ner'])
                    for k in m.keys():
                        for v in m[k]:
                            content = content.replace(v, "##"+v+"##"+k)
                    prompt.append({
                        "role": "assistant",
                        "content": content})

                prompt.append({
                    "role": "assistant", "content": d["sentence"]
                })

                s = getGpt(GPT35T, prompt)[0]
                # print("----------",prompt,"\n---------",s)
                # 正则表达式，捕捉@@和##之间的内容，及##后面的大写单词
                pattern = r'##(.*?)##(\w+)'

                # 使用正则表达式匹配所有符合模式的片段
                matches = re.findall(pattern, s)

                # 初始化一个映射（map），即字典
                parsed_map = {}

                # 遍历所有匹配
                for match in matches:
                    # match[0] 是 @@ 和 ## 之间的内容，match[1] 是 ## 后面的大写单词
                    key = match[1]
                    value = match[0]

                    # 如果键（即大写单词）已经在字典中，则将值添加到对应的数组中
                    # 否则，为这个键创建一个新数组，并添加值
                    if key in parsed_map:
                        parsed_map[key].append(value)
                    else:
                        parsed_map[key] = [value]

                # 输出解析后的映射（map）
                # print(parsed_map)
                # print("------------------/n",s,parsed_map,d["ner"])

                self.addResults(parsed_map, d["ner"])
                return {"sentence": d["sentence"], "label": d["ner"], "predict": parsed_map}
            except Exception as ex:
                print(f"An exception occurred: {ex},res:{s}")
                retries += 1
                if retries == 5:
                    print("Maximum retry limit reached. Failing with an error.")
                    return {}
                print(f"Retrying... {retries} attempts left.")
    def record(self,input :Any):
        # 序列化input为JSON格式字符串
        json_data = json.dumps(input, ensure_ascii=False, indent=4)
        # 将JSON数据写入到指定的文件路径
        with open(self.savePath, 'w', encoding='utf-8') as f:
            f.write(json_data)

    def getPRF1(self) -> Union[float, float, float]:
        if self.TP + self.FP==0 or self.TP + self.FN==0:
            return 0,0,0
        P = self.TP / (self.TP + self.FP)
        R = self.TP / (self.TP + self.FN)
        if P+R==0:
            return 0,0,0
        F1 = 2 * P * R / (P + R)
        return P, R, F1


if __name__ == '__main__':
    ner = MyNERExperience('data/test.txt', './result.txt',30)
    ner.run()
