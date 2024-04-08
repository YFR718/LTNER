import pickle

from pymilvus import MilvusClient

from common.chatgpt.gpt import embedding, ADA3Small, ADA2, ADA3Large, ADA3SmallWnut

client = MilvusClient(
    uri="xxx",  # 从控制台获取的集群公网地址
    token="xxx"  # 创建集群时指定的用户名和密码
)


ada3small_sentecnce   = {}
with open('data/test_embeddings.pkl', 'rb') as file:
    ada3small_sentecnce = pickle.load(file)


def search(text,name,model,limit,fields):
    emb = embedding(text,model)
    res = client.search(
        collection_name=name,
        data=[emb],
        limit=limit,
        output_fields=fields
    )
    return res[0]

def searchSentence(text,name,model,limit,fields):
    emb = ada3small_sentecnce[text]
    res = client.search(
        collection_name=name,
        data=[emb],
        limit=limit,
        output_fields=fields
    )
    return res[0]


def insert(datas,collection):
    res = client.insert(collection, data=datas)
    print("finish",len(res))
    client.flush(collection_name=collection)


if __name__ == '__main__':
    res = search("Rating ","conll003baseline",ADA3Small,10,["ner","sentence"])
    print(res[0]['entity']['ner'])