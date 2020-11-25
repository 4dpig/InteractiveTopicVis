from enum import Enum, unique
import itertools
import pandas as pd
import numpy as np
from scipy import sparse
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import zhconv


# 添加一个One-Hot表示
# 其实就是词频矩阵的0-1表示，出现1，未出现0

@unique
class CorpusLanguage(Enum):
    """
    语料库语言枚举类
    """
    English = 1
    Chinese = 2


class TextPreprocessing:
    """文本预处理类
    支持语言：
        中文
        英文
    初始化参数：
        documents：语料库，即某一类内容的若干文档的列表，每个文档用字符串来表示字符串
    """

    # 英文的nltk分词器对象，自定义了分词正则
    englishTokenizer = RegexpTokenizer(r"""
        (?x)                    # 设置以编写较长的正则条件
        (?:[A-Z]\.)+            # 缩略词 
        |\$?\d+(?:\.\d+)?%?     # 货币、百分数
        |\w+(?:[-']\w+)*        # 用连字符链接的词汇
        |\.\.\.                 # 省略符号 
        |(?:[.,;"'?():-_`])     # 特殊含义字符 
        """)

    # 英文stopwords
    with open('resource/stopwords/english.txt', 'r', encoding='UTF-8') as sw_file:
        englishStopwords = [stopword.rstrip() for stopword in sw_file.readlines()]
        englishStopwords.append(' ')

    # 中文停用词
    with open('resource/stopwords/chinese.txt', 'r', encoding='UTF-8') as sw_file:
        chineseStopwords = [stopword.rstrip() for stopword in sw_file.readlines()]
        chineseStopwords.append(' ')

    # scikit中提供的tfidf类，可以将词频矩阵转为tfidf矩阵
    tfidfTransformer = TfidfTransformer()

    def __init__(self, language: CorpusLanguage, documents, min_word_num_in_doc=3):
        # 提供的属性
        self.language = language
        self.docs = documents
        # 预处理后的文档至少要含有的单词数目，不然就会被过滤掉
        self.min_word_num_in_doc = min_word_num_in_doc
        self.docExcerptList = []
        self.docsWithList = []
        self.docsWithString = []
        self.bagWords = []
        self.wordFrequencyList = []
        self.tf = None
        self.tfidfDW = None
        self.tfidfWD = None
        self.wordDict = None

    def start(self):
        if self.language == CorpusLanguage.English:
            self.english_split_words()
        else:
            self.chinese_split_words()

        self.feature_extraction()

    def english_split_single_doc(self, doc):
        """
        # 空白字符全转为空格
        doc = re.sub(r'\s', ' ', doc)
        # 过滤非ASCII可见字符（32-126）
        doc = re.sub(r'[^\x20-\x7e]', '', doc)
        # 英文全转小写
        doc = doc.lower()
        # 分词
        words = self.englishTokenizer.tokenize(doc)
        # 去除停用词
        words = [word for word in words if word not in self.englishStopwords]
        """
        # 英文全转小写
        doc = doc.lower()
        # 分词
        words = self.englishTokenizer.tokenize(doc)
        # 去除停用词
        words = [word for word in words if word not in self.englishStopwords]

        return words

    def chinese_split_single_doc(self, doc):
        """
        # 英文全转小写
        doc = doc.lower()
        # 中文汉字全转简体
        doc = zhconv.convert(doc, 'zh-cn')
        # 非中文字符、英文字母、数字、连字符全转为空格
        doc = re.sub(r'((?![-a-z0-9\u4E00-\u9FA5]+).)+', ' ', doc)
        # 分词
        words = jieba.lcut(doc)
        # 去除停用词
        words = [word for word in words if word not in self.chineseStopwords]
        """

        # 非中文字符全转为空格
        doc = re.sub(r'[^\u4E00-\u9FA5]+', ' ', doc)
        # 中文汉字全转简体
        doc = zhconv.convert(doc, 'zh-cn')
        # 分词
        words = jieba.lcut(doc)
        # 去除停用词
        words = [word for word in words if word not in self.chineseStopwords]

        return words

    def english_split_words(self):
        """
        英文分词
        由于不同库的输入要求可能不一样
        所以分词后的文档提供有字符串（单词用空格分割）和单词列表2种形式：
        [
            'hello how are you',
            'i am fine'
        ]

        [
            ['hello', 'how', 'are', 'you'],
            ['i', 'am', 'fine']
        ]
        """

        di = 0
        while di < len(self.docs):
            doc = self.docs[di]
            words = self.english_split_single_doc(doc)

            # 添加进分词后的documents
            if len(words) >= self.min_word_num_in_doc:
                self.docExcerptList.append(
                    (doc[:20] + '...' if len(doc) > 20 else doc)
                )
                self.docsWithList.append(words)
                self.docsWithString.append(' '.join(words))
                di += 1
            else:
                del self.docs[di]

    def chinese_split_words(self):
        """
        中文分词
        """
        self.docsWithList = []
        self.docsWithString = []

        di = 0
        while di < len(self.docs):
            doc = self.docs[di]
            words = self.chinese_split_single_doc(doc)

            # 添加进分词后的documents
            if len(words) >= self.min_word_num_in_doc:
                self.docExcerptList.append(
                    (doc[:20] + '...' if len(doc) > 20 else doc)
                )
                self.docsWithList.append(words)
                self.docsWithString.append(' '.join(words))
                di += 1
            else:
                del self.docs[di]

    def feature_extraction(self):
        """文档特征提取
        计算以下内容：
            词袋模型：即语料库中所有文档包含的单词的集合
            假设文档数dn，词袋中单词数wn
            词频矩阵tf(dn x wn)： tf(i, j) = 文档i中单词j的出现次数
            词频-逆向文档频率tf-idf(dn x wn):
        """

        '''
        vectorizer = CountVectorizer()
        tf = vectorizer.fit_transform(self.docsWithString)
        fwords = vectorizer.get_feature_names()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(tf).toarray()
        '''

        # 计算词袋（即去重）
        dn = len(self.docs)
        self.bagWords = list(set(itertools.chain(*self.docsWithList)))
        wn = len(self.bagWords)

        # 建立 <单词，编号（索引值）>的字典
        self.wordDict = dict(zip(self.bagWords, range(0, wn)))

        # 计算词频和文档-词频矩阵
        self.wordFrequencyList = [0] * wn
        # self.tf = np.zeros((dn, wn), dtype=np.int)
        # 试一下scipy稀疏矩阵
        self.tf = sparse.lil_matrix((dn, wn), dtype=np.int)
        for di, doc in enumerate(self.docsWithList):
            for word in doc:
                wi = self.wordDict[word]
                self.tf[di, wi] += 1
                self.wordFrequencyList[wi] += 1

        # 由词频矩阵来计算tfidf（默认已进行l2标准化）
        self.tfidfDW = self.tfidfTransformer.fit_transform(self.tf).astype(np.float32)
        # 有些算法需要的输入数据是单词作为行，文档作为列
        self.tfidfWD = self.tfidfDW.transpose()

    def get_new_doc_tfidf(self, doc):
        """
        在此语料库的背景下，
        获取单一文档的tfidf矩阵
        """
        # 分词
        words = None
        if self.language == CorpusLanguage.Chinese:
            words = self.chinese_split_single_doc(doc)
        else:
            words = self.english_split_single_doc(doc)

        # 计算tf-idf
        wn = len(self.bagWords)
        tf = sparse.lil_matrix((1, wn), dtype=np.int)
        for word in words:
            # 如果此新文档中有语料库中未出现的单词，则忽略
            if word in self.wordDict:
                wi = self.wordDict[word]
                tf[0, wi] += 1

        # 由词频矩阵来计算tfidf（默认已进行l2标准化）
        tfidfDW = self.tfidfTransformer.fit_transform(tf).astype(np.float32)
        # 返回转置（wn x 1)
        return tfidfDW.transpose()
