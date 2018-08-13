import sys
from collections import Counter

import  numpy as np
import tensorflow.contrib.keras as kr


def batch_iter(title, word, label, batch_size=128):
    """生成批次数据"""
    data_len = len(title)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len - 1))
    title_shuffle = title[indices]
    word_shuffle = word[indices]
    label_shuffle = label[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield title_shuffle[start_id:end_id], word_shuffle[start_id:end_id], label_shuffle[start_id:end_id]


def read_file(titleFile, wordFile, labelFile):
    """:return title, words, labels列表"""
    titles, words, labels = [], [], []
    with open(titleFile,mode='r',encoding='utf-8', errors='ignore') as title:
        for line in title:
            try:
                titleProps = line.strip().split(' ')
                if titleProps:
                    titles.append(titleProps)
            except:
                pass

    with open(wordFile, mode='r', encoding='utf-8', errors='ignore') as word:
        for line in word:
            try:
                wordTemp = line.strip().split(' ')
                if wordTemp:
                    words.append(wordTemp)
            except:
                pass

    with open(labelFile, mode='r', encoding='utf-8', errors='ignore') as label:
        for line in label:
            try:
               labels.append(line.strip())
            except:
                pass
    return titles, words, labels

def read_title(titleFile):
    titles = []
    with open(titleFile,mode='r',encoding='utf-8', errors='ignore') as title:
        for line in title:
            try:
                titleProps = line.strip().split(' ')
                if titleProps:
                    titles.append(titleProps)
            except:
                pass
    return titles

def build_vocab(title_dir, vocab_dir, vocab_size = 100000):
    """:param 根据title props训练集构建词汇表"""
    data_train = read_title(title_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    open(vocab_dir, mode='w',encoding='utf-8').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """:param 读取词汇表"""
    with open(vocab_dir,mode='r',encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def read_category():
    """:param 读取两个分类"""
    categories = ['1','0']
    cat_to_ids = dict(zip(categories, range(len(categories))))
    return categories, cat_to_ids

def to_words(content, words):
    return ''.join(words[x] for x in content)

def process_file(titleFile, wordFile, labelFile, word_to_id, cat_to_id):
    """:return 转换成id 的 2D Tensor"""
    titles, words, labels = read_file(titleFile, wordFile, labelFile)
    # print(labels)
    # print(cat_to_id)
    title_id, word_id, label_id = [], [], []
    for i in range(len(titles)):
        length = []
        for x in titles[i]:
            if x in word_to_id:
               length.append(word_to_id[x])
            else:
               length.append(0)
        title_id.append(length)

    for i in range(len(words)):
        length = []
        for x in words[i]:
            if x in word_to_id:
               length.append(word_to_id[x])
            else:
               length.append(0)
        word_id.append(length)
        if labels[i] == '(null)' :labels[i] = '0'
        label_id.append(cat_to_id[labels[i]])
    title_pad = kr.preprocessing.sequence.pad_sequences(title_id)
    word_pad = kr.preprocessing.sequence.pad_sequences(word_id)
    label_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    return title_pad, word_pad, label_pad




#
#
#
# def process_title(titleFile, word_to_id):
#     """"""
#     titles = read_title(titleFile)
#
#     title_id, word_id, label_id = [], [], []
#     for i in range(len(titles)):
#         length = []
#         for x in titles[i]:
#             if x in word_to_id:
#                length.append(word_to_id[x])
#             else:
#                 length.append(0)
#         print(title_id)
#         title_id.append(length)
#     title_pad = kr.preprocessing.sequence.pad_sequences(title_id)
#     return title_pad
#
#
#
#
#
#
#
#
#
#
# title,_, label = read_file("/Users/hongkangjie/Downloads/tesla_title.txt","/Users/hongkangjie/Downloads/tesla_word.txt", "/Users/hongkangjie/Downloads/tesla_label.txt")
# # print(title)
#
# # build_vocab("/Users/hongkangjie/Downloads/tesla_test.txt","/Users/hongkangjie/Downloads/tesla_vocab.txt")
# _, word_to_id = read_vocab("/Users/hongkangjie/Downloads/tesla_vocab.txt")
# print(word_to_id)
# t = process_title("/Users/hongkangjie/Downloads/tesla_test.txt",word_to_id)
# print(t)