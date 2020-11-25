import time
from TextPreprocessing import CorpusLanguage, TextPreprocessing
from database import SessionLocal
import crud
from ssnmf2 import SSNMFTopicModel
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import jieba
import re
import zhconv
from umap.umap_ import UMAP
import numba as nb
from sklearn.manifold._t_sne import TSNE

text = '大sao 你b站id 被人拿去注册商标了 对方是重庆美家桓网络科技有限公司 ' \
      '申请 注册号 43554823 国际分类 35 申请日期 2020年1月3日 商标名称 ' \
      '徐大sao 申请人地址 重庆市九龙坡区含谷镇新营房村 ' \
      '环球锦标建材交易市场 a区7-10至11号 这是能查到的别瞎喷'

result = list(jieba.cut(text, use_paddle=True))




def test_callback(a, b):
    pass


@nb.jit(nopython=True)
def calculate(doc_x, doc_y):
    euclidean_distance = np.linalg.norm(doc_x - doc_y)
    if np.argmax(doc_x) == np.argmax(doc_y):
        return euclidean_distance * 0.4
    return euclidean_distance


session = SessionLocal()
comments = crud.get_bilibili_video_comments_by_vid(4, session)
session.close()

docs = [comment.text for comment in comments]

text_preprocessing = TextPreprocessing(CorpusLanguage.Chinese, docs)
text_preprocessing.start()

if False:
    ss_nmf = SSNMFTopicModel(test_callback, text_preprocessing.tfidfWD, 10, 20, 1000, 1e-4, 2020)
    time_start = time.time()
    ss_nmf.fit_original_nmf_cjlin_nnls()
    time_end = time.time()
    print(f'花费{time_end - time_start}秒')

    # 获取主题摘要
    bag_words = text_preprocessing.bagWords
    topic_summary_list = []
    for ti in range(10):
        top_n_word_index = np.argsort(ss_nmf.w[:, ti])[-3:]
        top_n_word_list = [bag_words[wi] for wi in top_n_word_index]
        print(f"主题{ti}摘要：{str(top_n_word_list)}")
else:
    lda = LatentDirichletAllocation(
        n_components=10,
        max_iter=1000,
        evaluate_every=1,
        perp_tol=1e-1,
        verbose=1
    )
    time_start = time.time()
    lda.fit_transform(text_preprocessing.tfidfDW)
    time_end = time.time()
    print(f"实际迭代次数：{lda.n_iter_}")
    print(f'花费{time_end - time_start}秒')
    # 获取主题摘要
    bag_words = text_preprocessing.bagWords
    topic_summary_list = []
    for ti in range(10):
        top_n_word_index = np.argsort(lda.components_[ti, :])[-3:]
        top_n_word_list = [bag_words[wi] for wi in top_n_word_index]
        print(f"主题{ti}摘要：{str(top_n_word_list)}")
"""
# 开始tsne
time_start = time.time()
if False:
    # 不用自定义距离函数：32.9秒
    # 使用自定义距离函数： 47.9秒
    tsne = TSNE(
        n_components=3,
        metric=calculate
    )
    doc_coordinate = tsne.fit_transform(ss_nmf.h.T)
else:
    # 不用自定义距离函数：14.4秒
    # 使用自定义距离函数： 18.3秒
    umap = UMAP(
        n_neighbors=15,
        n_components=3,
        metric=calculate,
        min_dist=0.1
    )
    doc_coordinate = umap.fit_transform(ss_nmf.h.T)
time_end = time.time()
print(f'花费{time_end - time_start}秒')
"""

