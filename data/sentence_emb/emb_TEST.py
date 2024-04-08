import pickle
from common.chatgpt.gpt import embeddings, ADA3Small
from data.loader.mapLoader import load

#  1. 读取data
data = load('../test.txt')
m = {}
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
    # 2.2 保存text2emb map

    for i in range(len(datas)):
        m[datas[i]['sentence']] = embs[i].embedding

with open('../test_embeddings.pkl', 'wb') as file:
    pickle.dump(m, file)








