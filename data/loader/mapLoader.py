def load(filepath):
    natural_language_sentences = []  # 用来存储转换后的自然语言标注句子
    current_words = []  # 当前句子的单词列表
    current_ner_tags = {}  # 当前句子中的NER标签字典

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("-DOCSTART-") or line.strip() == '':
                if current_words:
                    sentence_text = ' '.join(current_words)
                    natural_language_sentences.append({"sentence": sentence_text, "ner": current_ner_tags})
                    current_words = []
                    current_ner_tags = {}
                continue

            parts = line.strip().split()
            if parts:  # 确保不是空行
                word = parts[0]
                ner_tag = parts[-1]

                current_words.append(word)  # 添加单词到当前句子

                if ner_tag.startswith("B-"):
                    label = ner_tag.split("-")[-1]
                    if label not in current_ner_tags:
                        current_ner_tags[label] = []
                    current_ner_tags[label].append(word)
                elif ner_tag.startswith("I-") and label in current_ner_tags:
                    current_ner_tags[label][-1] += f" {word}"  # 将单词添加到最后一个实体中

    # 处理最后一个句子（如果存在）
    if current_words:
        sentence_text = ' '.join(current_words)
        natural_language_sentences.append({"sentence": sentence_text, "ner": current_ner_tags})

    return natural_language_sentences


if __name__ == '__main__':
    # 指定CoNLL-2003格式的NER数据集的文本文件路径
    conll_data_filepath = '../../data/CoNLL003_ENG/valid.txt'

    # 调用函数处理数据并打印结果
    natural_language_output = load(conll_data_filepath)

    # 这一行是为了美化输出，方便查看
    import json

    print(json.dumps(natural_language_output, indent=2, ensure_ascii=False))

    print(type(natural_language_output))
    print(len(natural_language_output))
    print(natural_language_output[0])