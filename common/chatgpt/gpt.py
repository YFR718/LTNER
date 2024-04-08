import time
from openai import OpenAI
GPT4T0125 = "gpt-4-0125-preview"
GPT4T1106 = "gpt-4-1106-preview"
GPT4T0613 = "gpt-4-0613"
# gpt3.5
GPT35T = "gpt-3.5-turbo-0125"
GPT35T0125 = "gpt-3.5-turbo-0125"
GPT35T1106 = "gpt-3.5-turbo-1106"
GPT35T0613 = "gpt-3.5-turbo-0613"



ADA2 = "text-embedding-ada-002"
ADA3Small = "text-embedding-3-small"
ADA3SmallWnut = "text-embedding-3-small-wnut"
ADA3Large = "text-embedding-3-large"


client = OpenAI(
    api_key = "xxxx"
)


# 创建一个锁对象
import threading

lock = threading.Lock()

prompt_tokens,completion_tokens = 0,0

def getGpt(model,prompt):
    global prompt_tokens,completion_tokens
    completion = client.chat.completions.create(
        model=model,
        messages= prompt,
        temperature=0
    )
    # print(completion)
    # 获取锁
    lock.acquire()
    try:
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens
        # print("usage:",prompt_tokens, completion_tokens)
    finally:
        # 释放锁
        lock.release()


    return completion.choices[0].message.content, completion.usage.total_tokens


# 编码一条数据
def embedding(text,model):
    while True:
        try:
            response = client.embeddings.create(
                input=text,
                model= model
            )
            return response.data[0].embedding
        except Exception as e:
            # 如果发生错误，打印错误信息
            print(f"Failed: {e}")
            break
    return response.data[0]



def embeddings(text,model):
    while True:
        try:
            response = client.embeddings.create(
                input=text,
                model= model
            )

            return response.data
        except Exception as e:
            # print(response)
            # 如果发生错误，打印错误信息
            print(f"Failed: {e}")
            time.sleep(1)
    return response.data



if __name__ == '__main__':
    prompt = [{"role":"user","content":"三个西瓜怎么分给四个孙子？"}]
    print(getGpt(GPT4T1106,prompt)[0])
    # embs = embeddings(["hello","hi","hood"],ADA3Small)
    # for e in embs:
    #     print(e)

    # embs = embeddings(["hood", "hi","hello"], ADA3Small)
    # for e in embs:
    #     print(e)