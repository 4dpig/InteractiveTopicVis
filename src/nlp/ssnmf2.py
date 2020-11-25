"""
半监督nmf的实现。
使用非负最小二乘法来实现nnls
"""
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import numba as nb
import pandas as pd
import streamlit as st
from scipy.optimize import nnls
from database import SessionLocal
import crud
from cjlin import nmf
import time
from TSNEWithCallback import TSNEWithCallback
import umap
import plotly.express as px
from typing import Callable


def squared_norm(x):
    """
    计算矩阵x的元素的平方和，
    比norm(x) ** 2更快
    """
    x = np.ravel(x, order='K')
    return np.dot(x, x)


def squared_norm_sparse(x):
    return sparse.linalg.norm(x) ** 2


class SSNMFTopicModel:
    def __init__(self,
                 callback_set_nmf_training_progress: Callable[[int, float], None],
                 a: np.ndarray,
                 k: int,
                 min_iter: int = 10,
                 max_iter: int = 10000,
                 tol: float = 1e-4,
                 seed: int = 2020,
                 ):
        """
        参数：
        a：词项文档矩阵，行单词，列文档
        m：词数
        n：文档数
        k：假设的主题数量
        maxIter：最大迭代次数
        tol：停止迭代的容错值
        seed：使用随机数初始化W、H时选取的种子
        rng：随机数生成器对象
        """
        self.callback_set_nmf_training_progress = callback_set_nmf_training_progress
        self.a = a
        self.a_dok = a.todok()
        self.m, self.n = self.a.shape
        self.k = k
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(seed)

        # 使用随机数初始化矩阵w和h
        self.random_init_wh()

        # wr, mw, hr, mh均初始化为0矩阵
        self.wr = np.zeros((self.m, self.k), dtype=np.float32)
        self.mw = np.zeros((self.k, self.k), dtype=np.float32)
        self.hr = np.zeros((self.k, self.n), dtype=np.float32)
        self.mh = sparse.csr_matrix((self.n, self.n), dtype=np.float32)

        # ik作为单位矩阵
        self.ik = np.identity(self.k, dtype=np.int)

        # 计算误差时对h按列分块的step步长
        # 分块后的矩阵是m x step (float32)
        # 按照最大100M来计算
        self.column_step = int(1e8 / (4 * self.m))
        self.is_block = True if self.n > self.column_step else False

    def random_init_wh(self):
        """
        使用RandomState产生符合标准正态分布的随机矩阵，
        然后使用avg来缩放（mean()计算矩阵中元素的平均值），
        最后用abs来过滤掉负值
        """
        avg = np.sqrt(self.a.mean() / self.k)

        self.w = avg * self.rng.randn(self.m, self.k).astype(np.float32)
        np.abs(self.w, self.w)

        self.h = avg * self.rng.randn(self.k, self.n).astype(np.float32)
        np.abs(self.h, self.h)

    def error_new_doc(self, a, h):
        a_minus_wh = a - np.dot(self.w, h)
        return squared_norm(a_minus_wh)

    def error_original_nmf(self):
        """
        计算原始nmf的损失函数值
        """
        if self.is_block:
            error_sum = 0
            ci_start = 0
            ci_end = 0

            while ci_end < self.n:
                ci_start = ci_end
                ci_end += self.column_step
                if ci_end > self.n:
                    ci_end = self.n

                h_block = self.h[:, ci_start: ci_end]
                wh_block_dot = np.dot(self.w, h_block)
                error_block = self.a[:, ci_start: ci_end] - wh_block_dot
                error_sum += (squared_norm(error_block))

            return error_sum
        else:
            return squared_norm(self.a - np.dot(self.w, self.h))

    def error_semi_supervised_nmf(self):
        """
        计算半监督nmf的损失函数值
        """
        error1 = self.error_original_nmf()

        matrix2 = np.dot(self.w - self.wr, self.mw)
        error2 = squared_norm(matrix2)

        matrix3 = (self.h - self.hr) * self.mh
        error3 = squared_norm(matrix3)

        return error1 + error2 + error3

    def get_initgrad(self):
        gradW = np.dot(self.w, np.dot(self.h, self.h.T)) - np.dot(self.a, self.h.T)
        gradH = np.dot(np.dot(self.w.T, self.w), self.h) - np.dot(self.w.T, self.a)
        initgrad = np.linalg.norm(np.r_[gradW, gradH.T])
        return initgrad

    def fit_new_doc(self, a):
        # 固定w，根据新a求h
        avg = np.sqrt(a.mean() / self.k)
        h = avg * self.rng.randn(self.k, 1).astype(np.float32)
        np.abs(h, h)

        # 设置初始误差
        tolh = max(0.001, self.tol)
        error_at_init = self.error_new_doc(a, h)
        print(error_at_init)
        self.error_array = [error_at_init]

        at = a.transpose()
        for i in range(self.max_iter):
            # 固定W求H
            h, _, iter_count = nmf.nlssubprob(a, self.w, h, tolh, 1000)
            if iter_count == 1:
                tolh *= 0.1

            error = self.error_new_doc(a, h)
            print(error)
            self.error_array.append(error)

            if i + 1 >= self.min_iter and \
                    np.abs(error - self.error_array[i]) / error_at_init <= self.tol:
                print(f'达到收敛条件，在第{i}次迭代后提前结束')
                break

        return h

    def fit_original_nmf_cjlin_nnls(self):
        """
        计算原始的nmf，损失函数：

        """
        tolw = max(0.001, self.tol)
        tolh = tolw
        error_at_init = self.error_original_nmf()
        self.error_array = [error_at_init]
        self.callback_set_nmf_training_progress(0, error_at_init)

        at = self.a.transpose()
        for i in range(self.max_iter):
            if i % 2 == 0:
                # 固定H求W
                self.w, _, iter_count = nmf.nlssubprob(at, self.h.T, self.w.T, tolw, 1000)
                self.w = self.w.T
                if iter_count == 1:
                    tolw *= 0.1

            else:
                # 固定W求H
                self.h, _, iter_count = nmf.nlssubprob(self.a, self.w, self.h, tolh, 1000)
                if iter_count == 1:
                    tolh *= 0.1

            error = self.error_original_nmf()
            print(error)
            self.error_array.append(error)
            self.callback_set_nmf_training_progress(i + 1, error)

            if i + 1 >= self.min_iter and \
                    np.abs(error - self.error_array[i]) / error_at_init <= self.tol:
                print(f'达到收敛条件，在第{i}次迭代后提前结束')
                break

        # self.result_column_normalized()

    def fit_semi_supervised_nmf_cjlin_nnls(self):
        """
        计算半监督的nmf，损失函数：
        """
        tolw = max(0.001, self.tol)
        tolh_list = [tolw] * self.n
        error_at_init = self.error_semi_supervised_nmf()
        self.error_array = [error_at_init]
        self.callback_set_nmf_training_progress(0, error_at_init)

        vstack_ht_mw = np.vstack((self.h.T, self.mw))
        vstack_at_hstack_mw_wrt = sparse.vstack(
            (self.a.transpose(), np.dot(self.mw, self.wr.T))
        )

        for i in range(self.max_iter):
            if i % 2 == 0:
                # 固定H求W
                vstack_ht_mw[0:self.n, 0:self.k] = self.h.T
                '''
                vstack_ht_mw = np.vstack((self.h.T, self.mw))
                vstack_at_hstack_mw_wrt = np.vstack(
                    (self.a.T, np.dot(self.mw, self.wr.T))
                )
                '''

                self.w, _, iter_count = nmf.nlssubprob(vstack_at_hstack_mw_wrt,
                                                       vstack_ht_mw, self.w.T,
                                                       tolw, 1000)
                self.w = self.w.T
                if iter_count == 1:
                    tolw *= 0.1
                print(f'迭代次数{iter_count}')
            else:
                # 固定W求H
                for ci in range(self.n):
                    vstack_w_mhi_ik = np.vstack((self.w, self.mh[ci, ci] * self.ik))

                    vstack_aci_mhi_dhi_hrci = sparse.vstack(
                        (
                            self.a[:, ci].reshape((self.m, 1)),
                            (self.mh[ci, ci] * self.hr[:, ci]).reshape(
                                (self.k, 1))
                        )
                    )
                    hci, _, iter_count = nmf.nlssubprob(
                        vstack_aci_mhi_dhi_hrci,
                        vstack_w_mhi_ik,
                        self.h[:, ci].reshape((self.k, 1)),
                        tolh_list[ci],
                        1000
                    )

                    self.h[:, ci] = hci.reshape(self.k)
                    if iter_count == 1:
                        tolh_list[ci] *= 0.1

                """
                # 更新DH
                dh_lil = self.dh.tolil()
                for j in range(self.n):
                    if self.mh[j, j] == 0:
                        dh_lil[j, j] = 0
                    else:
                        hri = self.hr[:, j]
                        # 这里向量乘向量不需要reshape
                        # 因为numpy中，np.dot(v1, v2)
                        # 就相当于v1.T * v2 (v1∈(n, 1), v2∈(n, 1))
                        dh_lil[j, j] = (
                                np.dot(hri, self.h[:, j]) /
                                np.dot(hri, hri)
                        )

                self.dh = dh_lil.tocsr()
                """

            error = self.error_semi_supervised_nmf()
            print(error)
            self.error_array.append(error)
            self.callback_set_nmf_training_progress(i + 1, error)

            if i + 1 >= self.min_iter and \
                    np.abs(error - self.error_array[i]) / error_at_init <= self.tol:
                print(f'达到收敛条件，在第{i}次迭代后提前结束')
                break

        # self.result_column_normalized()

    def result_column_normalized(self):
        """
        对W、H的结果进行调整，
        使得W列归一化
        """

        dw = np.zeros((self.k, self.k), dtype=np.float32)
        for i in range(self.k):
            dw[i, i] = sum(self.w[:, i])

        dw_inv = np.linalg.inv(dw)

        self.w = np.dot(self.w, dw_inv)
        self.h = np.dot(dw, self.h)

        print(self.error_original_nmf())


def test_callback(a, b):
    pass


@nb.jit(nopython=True)
def custom_doc_distance_callback(doc_x, doc_y):
    """
    在sklearn的t-sne算法中，
    用于自定义计算文档（样本）之间成对距离的函数
    """
    euclidean_distance = np.linalg.norm(doc_x - doc_y)
    if np.argmax(doc_x) == np.argmax(doc_y):
        return euclidean_distance * 0.4
    return euclidean_distance


def tsne_callback(current_iter, doc_coordinate):
    print("TSNE回调", current_iter)
    print(doc_coordinate.shape)


if __name__ == '__main__':
    from TextPreprocessing import CorpusLanguage, TextPreprocessing

    session = SessionLocal()
    comments = crud.get_bilibili_video_comments_by_vid(6, session)
    session.close()

    docs = [comment.text for comment in comments]

    text_preprocessing = TextPreprocessing(CorpusLanguage.Chinese, docs)
    text_preprocessing.start()
    ss_nmf = SSNMFTopicModel(test_callback, text_preprocessing.tfidfWD, 10, 20, 1000, 1e-5, 2020)

    time_start = time.time()
    ss_nmf.fit_original_nmf_cjlin_nnls()
    time_end = time.time()
    print(f'花费{time_end - time_start}秒')
    print('-----------------------------------------------')

    df = pd.DataFrame(
        ss_nmf.w,
        columns=[f'主题{ti}' for ti in range(ss_nmf.k)],
        index=text_preprocessing.bagWords
    )

    df2 = pd.DataFrame(
        ss_nmf.h.T,
        columns=[f'主题{ti}' for ti in range(ss_nmf.k)],
        index=text_preprocessing.docs,
    )

    st.dataframe(df)
    st.dataframe(df2)

    word_dict = text_preprocessing.wordDict

    '''
    # 对主题0进行关键词优化
    ss_nmf.mw[0, 0] = 1
    ss_nmf.wr[:, 0] = ss_nmf.w[:, 0]
    ss_nmf.wr[word_dict['泷'], 0] = 5
    ss_nmf.wr[word_dict['三叶'], 0] = 5
    ss_nmf.wr[word_dict['前辈'], 0] = 1
    ss_nmf.wr[word_dict['写'], 0] = 0
    ss_nmf.wr[word_dict['年'], 0] = 0
    '''

    '''
    # 将主题0和主题1合并为1个主题
    t1 = 0
    t2 = 1

    ss_nmf.h[t1, :] += ss_nmf.h[t2, :]
    ss_nmf.h = np.delete(ss_nmf.h, t2, axis=0)
    ss_nmf.hr = np.delete(ss_nmf.hr, t2, axis=0)

    ss_nmf.wr[:, t1] = (ss_nmf.w[:, t1] + ss_nmf.w[:, t2]) / 2
    ss_nmf.wr = np.delete(ss_nmf.wr, t2, axis=1)
    ss_nmf.w = np.delete(ss_nmf.w, t2, axis=1)
    
    ss_nmf.k -= 1
    ss_nmf.ik = np.identity(ss_nmf.k)
    ss_nmf.mw = np.zeros((ss_nmf.k, ss_nmf.k))
    ss_nmf.mw[t1, t1] = 1
    '''

    '''
    # 将主题0拆分为2个主题
    ti = 0
    new_topic_column = ss_nmf.w[:, ti]

    ss_nmf.w = np.insert(ss_nmf.w, ti+1, values=new_topic_column, axis=1)
    ss_nmf.wr[:, ti] = new_topic_column
    ss_nmf.wr = np.insert(ss_nmf.wr, ti+1, values=new_topic_column, axis=1)

    ss_nmf.h = np.insert(ss_nmf.h, ti+1, values=ss_nmf.h[ti, :], axis=0)
    ss_nmf.hr = np.insert(ss_nmf.hr, ti+1, values=np.zeros(ss_nmf.n), axis=0)

    ss_nmf.wr[word_dict['泷'], ti] = 5
    ss_nmf.wr[word_dict['三叶'], ti] = 5
    ss_nmf.wr[word_dict['时间'], ti] = 0
    ss_nmf.wr[word_dict['年'], ti] = 0
    ss_nmf.wr[word_dict['三年'], ti] = 0

    ss_nmf.wr[word_dict['泷'], ti+1] = 0
    ss_nmf.wr[word_dict['三叶'], ti+1] = 0
    ss_nmf.wr[word_dict['时间'], ti+1] = 3
    ss_nmf.wr[word_dict['年'], ti+1] = 3
    ss_nmf.wr[word_dict['三年'], ti+1] = 3
    
    ss_nmf.k += 1
    ss_nmf.ik = np.identity(ss_nmf.k)
    ss_nmf.mw = np.zeros((ss_nmf.k, ss_nmf.k))
    ss_nmf.mw[ti, ti] = 1
    ss_nmf.mw[ti+1, ti+1] = 1
    '''

    '''
    # 文档诱导的主题创建
    induce_doc_index_list = [8, 14, 27]
    for di in induce_doc_index_list:
        print(text_preprocessing.docs[di])

    ss_nmf.h = np.insert(ss_nmf.h, ss_nmf.k, values=np.zeros(ss_nmf.n), axis=0)
    ss_nmf.hr = np.insert(ss_nmf.hr, ss_nmf.k, values=np.zeros(ss_nmf.n), axis=0)
    ss_nmf.hr[-1, induce_doc_index_list] = 1
    ss_nmf.w = np.insert(ss_nmf.w, ss_nmf.k, values=np.zeros(ss_nmf.m), axis=1)
    ss_nmf.wr = np.insert(ss_nmf.wr, ss_nmf.k, values=np.zeros(ss_nmf.m), axis=1)

    ss_nmf.k += 1
    ss_nmf.ik = np.identity(ss_nmf.k)
    ss_nmf.mw = np.zeros((ss_nmf.k, ss_nmf.k))
    ss_nmf.mh[induce_doc_index_list, induce_doc_index_list] = 100
    '''

    '''
    # 单词诱导的主题创建
    induce_word_index_list = [word_dict['日本'], word_dict['电影']]

    ss_nmf.w = np.insert(ss_nmf.w, ss_nmf.k, values=np.zeros(ss_nmf.m), axis=1)
    ss_nmf.wr = np.insert(ss_nmf.wr, ss_nmf.k, values=np.zeros(ss_nmf.m), axis=1)
    ss_nmf.wr[induce_word_index_list, ss_nmf.k] = 1
    ss_nmf.h = np.insert(ss_nmf.h, ss_nmf.k, values=np.zeros(ss_nmf.n), axis=0)
    ss_nmf.hr = np.insert(ss_nmf.hr, ss_nmf.k, values=np.zeros(ss_nmf.n), axis=0)

    ss_nmf.k += 1
    ss_nmf.ik = np.identity(ss_nmf.k)
    ss_nmf.mw = np.zeros((ss_nmf.k, ss_nmf.k))
    ss_nmf.mw[ss_nmf.k - 1, ss_nmf.k - 1] = 1


    time_start = time.time()
    ss_nmf.fit_semi_supervised_nmf_cjlin_nnls()
    time_end = time.time()
    print(f'花费{time_end - time_start}秒')

    df = pd.DataFrame(
        ss_nmf.w,
        columns=[f'主题{ti}' for ti in range(ss_nmf.k)],
        index=text_preprocessing.bagWords
    )

    df2 = pd.DataFrame(
        ss_nmf.h,
        index=[f'主题{ti}' for ti in range(ss_nmf.k)],
        columns=text_preprocessing.docs,
    )

    st.dataframe(df)
    st.dataframe(df2)
    '''

    time_start = time.time()
    # tsne可视化

    # 文档硬聚类，即将每个文档归类到与其关联最大的那个主题中
    doc_hard_clustering_topic = np.argmax(ss_nmf.h, axis=0).tolist()
    # 提前计算距离
    # 缩放比例
    '''
    scaling_ratio = 0.1
    distance_matrix = np.zeros((ss_nmf.n, ss_nmf.n))
    for i in range(ss_nmf.n):
        di_vector = ss_nmf.h[:, i]
        di_topic = doc_hard_clustering_topic[i]
        j = i + 1
        while j < ss_nmf.n:
            distance = np.linalg.norm(di_vector - ss_nmf.h[:, j])
            if di_topic == doc_hard_clustering_topic[j]:
                distance *= scaling_ratio
            distance_matrix[i, j] = distance_matrix[j, i] = distance
            j += 1
    '''

    t_sne = TSNEWithCallback(tsne_callback, 5, 3, metric=custom_doc_distance_callback,
                             random_state=2020)
    doc_coordinate = t_sne.fit_transform(ss_nmf.h.T)

    '''
    umap_reducer = umap.UMAP(
        n_components=3,
        metric=custom_doc_distance_callback,
        random_state=2020
    )
    doc_coordinate = umap_reducer.fit_transform(ss_nmf.h.T)
    '''

    time_end = time.time()
    print(f'花费{time_end - time_start}秒')
    doc_coordinate_df = pd.DataFrame(
        doc_coordinate,
        columns=['x', 'y', 'z'],
        index=text_preprocessing.docs
    )

    fig = px.scatter_3d(doc_coordinate_df, x='x', y='y', z='z')
    st.plotly_chart(fig)
