import json

from common.chatgpt.gpt import ADA3Small, embeddings
from common.vector import zilliz
from data.loader.mapLoader import load

# collection name
Collection = "test"

#  1. 读取data
data1 = load('../train.txt')
data2 = load('../valid.txt')
data = data1 + data2

# 2. 分批次处理
batch_size = 1000  # 每批处理1000个元素
for start_index in range(0, len(data), batch_size):
    # 计算批处理的结束索引
    end_index = min(start_index + batch_size, len(data))
    # 获取当前批次的数据
    datas = data[start_index:end_index]
    # 2.1 emb
    sentence = []
    for d in datas:
        sentence.append(d['sentence'])


    embs = embeddings(sentence, ADA3Small)
    # 2.2 上传zilliz
    inserts = []
    for i in range(len(datas)):
        inserts.append({
            "vector":embs[i].embedding,
            "ner":json.dumps(datas[i]['ner']),
            "sentence":datas[i]['sentence']
        })
    zilliz.insert(inserts,Collection)







