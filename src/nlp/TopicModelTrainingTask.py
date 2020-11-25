from enum import Enum, unique
from pydantic import BaseModel
from database import SessionLocal
from scipy import sparse
import numpy as np
import numba as nb
from umap.umap_ import UMAP
import crud
from TextPreprocessing import TextPreprocessing, CorpusLanguage
from ssnmf2 import SSNMFTopicModel
from typing import List, Dict
import time


@unique
class CorpusType(Enum):
    """
    语料库类型枚举类
    """
    BilibiliVideoComments = 1
    TakeawayReviews = 2
    OnlineShoppingReviews = 3
    ChineseLyrics = 4
    COVID19News = 5


class TextPreprocessingParams(BaseModel):
    corpus_type: CorpusType
    language: CorpusLanguage
    doc_min_word_num: int
    video_id: int = -1


class WordFrequency(BaseModel):
    text: str = ""
    value: int = 0


class TextPreprocessingProgress(BaseModel):
    status_code: int = 0
    message: str = "未开始"
    word_frequency_list: List[WordFrequency] = []


class NMFTrainingParams(BaseModel):
    topic_num: int
    min_iter: int
    max_iter: int
    tolerance: float
    dimension: int
    n_neighbors: int
    min_dist: float
    scaling_ratio: float
    random_seed: int = 2020


class SSNMFTrainingParams(BaseModel):
    min_iter: int
    max_iter: int
    tolerance: float
    n_neighbors: int
    min_dist: float
    scaling_ratio: float


class TopicKeywordOptimizationParams(BaseModel):
    ssnmf_training_params: SSNMFTrainingParams = None
    topic_id: int = 0
    word_new_weight_dict: Dict[int, float] = {}
    supervision_intensity: float = 0


class TopicSplitParams(BaseModel):
    ssnmf_training_params: SSNMFTrainingParams = None
    new_topic1_id: int = 0
    word_new_weight_dict1: Dict[int, float] = {}
    supervision_intensity1: float = 0
    new_topic2_id: int = 0
    word_new_weight_dict2: Dict[int, float] = {}
    supervision_intensity2: float = 0


class TopicMergeParams(BaseModel):
    ssnmf_training_params: SSNMFTrainingParams = None
    topic1_id: int = 0
    topic2_id: int = 0
    word_new_weight_dict: Dict[int, float] = {}
    supervision_intensity: float = 0


class KeywordInducedTopicCreateParams(BaseModel):
    ssnmf_training_params: SSNMFTrainingParams = None
    induce_word_weight_dict: Dict[int, float] = {}
    supervision_intensity: float = 0


class DocumentInducedTopicCreateParams(BaseModel):
    ssnmf_training_params: SSNMFTrainingParams = None
    induce_doc_relevancy_dict: Dict[int, float] = {}
    induce_doc_supervision_dict: Dict[int, float] = {}


# 前端可视化文档主题聚类时所需要的数据
class DocumentNodeData(BaseModel):
    doc_id: int = -1
    excerpt: str = ""
    x: float = -1
    y: float = -1
    z: float = -1


class TopicClusterData(BaseModel):
    topic_id: int = -1
    topic_summary: str = ""
    documents: List[DocumentNodeData] = []


class NMFTrainingProgress(BaseModel):
    status_code: int = 0
    current_iter: int = 0
    error_list: List[float] = []
    message: str = "未开始"


class UMAPProgress(BaseModel):
    status_code: int = 0
    topic_clusters: List[TopicClusterData] = []
    message: str = "未开始"


class DocumentDetails(BaseModel):
    doc_id: int = 0
    text: str = ""
    word_list: List[str] = []
    word_frequency_list: List[int] = []
    topic_id_list: List[int] = []
    topic_list: List[str] = []
    topic_summary_list: List[str] = []
    topic_relevancy_list: List[float] = []


class TopicDetails(BaseModel):
    topic_id: int = 0
    word_id_list: List[int] = []
    word_list: List[str] = []
    word_distribution_list: List[float] = []
    hard_clustering_doc_id_list: List[int] = []
    hard_clustering_doc_list: List[str] = []
    hard_clustering_doc_text_list: List[str] = []
    hard_clustering_doc_relevancy_list: List[float] = []


class KeywordInfo(BaseModel):
    word_id: int = 0
    word: str = ""
    word_weight: float = 0
    word_weight_percent: float = 0


class KeywordSearchResult(BaseModel):
    word_id_list: List[int] = []
    word_list: List[str] = []


class TopicKeywordInfo(BaseModel):
    topic_id: int = 0
    word_weight_sum: float = 0
    keyword_info_list: List[KeywordInfo] = []


class TopicMergeKeywordInfo(BaseModel):
    topic1_id: int = 0
    topic2_id: int = 0
    word_weight_sum: float = 0
    keyword_info_list: List[KeywordInfo] = []


class NewDocTopicDistributionPredictProgress(BaseModel):
    status_code: int = 0
    topic_summary_list: List[str] = []
    topic_relevancy_percent_list: List[float] = []


def custom_doc_distance_callback(scaling_ratio):
    """
    在UMAP降维算法中，
    用于自定义计算文档（样本）之间成对距离的函数
    """
    @nb.jit(nopython=True)
    def calculate(doc_x, doc_y):
        euclidean_distance = np.linalg.norm(doc_x - doc_y)
        if np.argmax(doc_x) == np.argmax(doc_y):
            return euclidean_distance * scaling_ratio
        return euclidean_distance

    return calculate


class TopicModelTrainingTask:
    def __init__(self, text_preprocessing_params: TextPreprocessingParams):
        self.text_preprocessing_params = text_preprocessing_params
        self.text_preprocessing_progress = TextPreprocessingProgress()
        self.text_preprocessing: TextPreprocessing = None
        self.nmf_training_progress: NMFTrainingProgress = None
        self.umap_progress: UMAPProgress = None
        self.topic_distribution_predict_progress = None
        self.ss_nmf: SSNMFTopicModel = None
        self.umap: UMAP = None
        self.doc_coordinate: np.ndarray = None

    def preprocessing(self):
        self.text_preprocessing_progress.status_code = 1
        self.text_preprocessing_progress.message = "读取数据库中"
        db = SessionLocal()

        # 根据不同的语料库读取不同的数据
        comments = []
        corpus_type = self.text_preprocessing_params.corpus_type
        if corpus_type == CorpusType.BilibiliVideoComments:
            comments = crud.get_bilibili_video_comments_by_vid(
                self.text_preprocessing_params.video_id, db
            )
        elif corpus_type == CorpusType.TakeawayReviews:
            comments = crud.get_takeaway_reviews(db)
        elif corpus_type == CorpusType.OnlineShoppingReviews:
            comments = crud.get_online_shopping_reviews(db)
        elif corpus_type == CorpusType.ChineseLyrics:
            comments = crud.get_chinese_lyrics(db)
        else:
            comments = crud.get_COVID19_news(db)

        if not comments:
            self.text_preprocessing_progress.status_code = -1
            self.text_preprocessing_progress.message = "读取数据库出错！"
            db.close()
            return

        docs = [comment.text for comment in comments]
        self.text_preprocessing = TextPreprocessing(
            self.text_preprocessing_params.language,
            docs,
            self.text_preprocessing_params.doc_min_word_num
        )

        self.text_preprocessing_progress.message = "预处理中"
        self.text_preprocessing.start()

        # 构造前端需要的词云数据格式
        bag_words = self.text_preprocessing.bagWords
        word_frequency_list = self.text_preprocessing.wordFrequencyList
        for wi, wf in enumerate(word_frequency_list):
            word_frequency = WordFrequency(
                text=bag_words[wi],
                value=wf
            )
            self.text_preprocessing_progress.word_frequency_list.append(word_frequency)

        self.text_preprocessing_progress.status_code = 2
        self.text_preprocessing_progress.message = "预处理完成"
        db.close()

    def callback_set_nmf_training_progress(self, current_iter: int, error_value: float) -> None:
        """
        回调：设置nmf训练状态信息
        """
        self.nmf_training_progress.current_iter = current_iter
        self.nmf_training_progress.error_list.append(error_value)
        self.nmf_training_progress.message = f"正在进行第{current_iter}次迭代"

    def calculate_tsne_distance_matrix(self, scaling_ratio):
        """
        计算T-SNE算法的输入距离矩阵
        """
        # 文档硬聚类，即将每个文档归类到与其关联最大的那个主题中
        doc_hard_clustering_topic = np.argmax(self.ss_nmf.h, axis=0).tolist()
        # 提前计算距离
        distance_matrix = np.zeros((self.ss_nmf.n, self.ss_nmf.n))
        for i in range(self.ss_nmf.n):
            di_vector = self.ss_nmf.h[:, i]
            di_topic = doc_hard_clustering_topic[i]
            j = i + 1
            while j < self.ss_nmf.n:
                distance = np.linalg.norm(di_vector - self.ss_nmf.h[:, j])
                if di_topic == doc_hard_clustering_topic[j]:
                    distance *= scaling_ratio
                distance_matrix[i, j] = distance_matrix[j, i] = distance
                j += 1

        return distance_matrix

    def start_nmf_and_umap(self, is_first_training: bool = True):
        # 开始NMF训练
        self.nmf_training_progress.status_code = 1
        if is_first_training:
            self.nmf_training_progress.message = "开始运行NMF"
            self.ss_nmf.fit_original_nmf_cjlin_nnls()
        else:
            self.nmf_training_progress.message = "开始运行半监督NMF"
            self.ss_nmf.fit_semi_supervised_nmf_cjlin_nnls()

        # 计算相关统计数据
        topic_num = self.ss_nmf.k
        doc_num = self.ss_nmf.n
        # 每个文档硬聚类到的主题
        self.doc_hard_clustering_topic = np.argmax(self.ss_nmf.h, axis=0).tolist()
        # 每个主题集群所包含的文档
        self.topic_cluster_doc_indexes_list = [[] for _ in range(topic_num)]
        for di in range(doc_num):
            self.topic_cluster_doc_indexes_list[self.doc_hard_clustering_topic[di]].append(di)
        # 按照从小到大的顺序
        for ti in range(topic_num):
            topic_relevancy_to_each_doc = self.ss_nmf.h[ti, :]
            self.topic_cluster_doc_indexes_list[ti].sort(
                key=lambda doc_index: topic_relevancy_to_each_doc[doc_index]
            )

        # 获取主题摘要
        bag_words = self.text_preprocessing.bagWords
        self.topic_summary_list = []
        for ti in range(topic_num):
            top_n_word_index = np.argsort(self.ss_nmf.w[:, ti])[-3:]
            top_n_word_list = [bag_words[wi] for wi in top_n_word_index]
            self.topic_summary_list.append(','.join(top_n_word_list))

        self.nmf_training_progress.status_code = 2
        self.nmf_training_progress.message = "训练完成！"

        # 开始UMAP降维
        # 重新设置进度
        self.umap_progress.topic_clusters = []
        self.umap_progress.status_code = 1
        self.umap_progress.message = "开始运行UMAP"
        start = time.time()
        doc_coordinate = self.umap.fit_transform(self.ss_nmf.h.T)
        end = time.time()
        print(f"花费{end - start}秒")
        self.umap_progress.topic_clusters = self.get_topic_cluster_data(doc_coordinate)
        self.umap_progress.status_code = 2
        self.umap_progress.message = "降维完成！"

        print("完成！")

    def nmf_training(self, nmf_training_params: NMFTrainingParams):
        # 创建训练状态信息对象
        self.nmf_training_progress = NMFTrainingProgress()
        self.umap_progress = UMAPProgress()

        # 创建SSNMF和UMAP对象
        self.ss_nmf = SSNMFTopicModel(
            self.callback_set_nmf_training_progress,
            self.text_preprocessing.tfidfWD,
            nmf_training_params.topic_num,
            min_iter=nmf_training_params.min_iter,
            max_iter=nmf_training_params.max_iter,
            tol=nmf_training_params.tolerance,
            seed=nmf_training_params.random_seed
        )

        """
        self.t_sne = TSNEWithCallback(
            update_progress_callback=self.callback_set_tsne_progress,
            update_progress_each_iter=10,
            n_components=3,
            perplexity=nmf_training_params.perplexity,
            learning_rate=nmf_training_params.learning_rate,
            n_iter=nmf_training_params.tsne_max_iter,
            metric=custom_doc_distance_callback(nmf_training_params.scaling_ratio),
            init="pca",
            random_state=nmf_training_params.random_seed
        )
        """
        print("参数值：")
        print(nmf_training_params.n_neighbors)
        print(nmf_training_params.min_dist)
        self.umap = UMAP(
            n_neighbors=nmf_training_params.n_neighbors,
            n_components=nmf_training_params.dimension,
            metric=custom_doc_distance_callback(nmf_training_params.scaling_ratio),
            min_dist=nmf_training_params.min_dist
        )

        # 开始训练
        self.start_nmf_and_umap()

    def reset_training_progress_and_params(self, params: SSNMFTrainingParams):
        # 重新创建训练状态信息对象
        self.nmf_training_progress = NMFTrainingProgress()
        self.umap_progress = UMAPProgress()

        # 重新设置NMF参数
        self.ss_nmf.min_iter = params.min_iter
        self.ss_nmf.max_iter = params.max_iter
        self.ss_nmf.tol = params.tolerance

        # 重新设置UMAP参数
        self.umap.n_neighbors = params.n_neighbors
        self.umap.min_dist = params.min_dist
        # 更新集群距离缩放比例值
        self.umap.metric = custom_doc_distance_callback(params.scaling_ratio)

    def topic_keyword_optimization(self, tko_params: TopicKeywordOptimizationParams):
        # 重新设置训练参数
        self.reset_training_progress_and_params(tko_params.ssnmf_training_params)

        # 设置主题关键词优化的半监督nmf相关参数、矩阵
        topic_id = tko_params.topic_id
        # 设置监督强度
        self.ss_nmf.mw[:] = 0
        self.ss_nmf.mw[topic_id, topic_id] = tko_params.supervision_intensity
        # wr参考矩阵的相应主题列先初始化和w一样
        self.ss_nmf.wr[:] = 0
        self.ss_nmf.wr[:, topic_id] = self.ss_nmf.w[:, topic_id]
        # 为有变动的单词设置新的权重
        for item in tko_params.word_new_weight_dict.items():
            word_id, new_weight = item
            self.ss_nmf.wr[word_id, topic_id] = new_weight

        # 重制hr，mh
        self.ss_nmf.hr[:] = 0
        dn = self.ss_nmf.n
        self.ss_nmf.mh = sparse.csr_matrix((dn, dn), dtype=np.float32)

        # 开始训练
        self.start_nmf_and_umap(is_first_training=False)

    def topic_split(self, ts_params: TopicSplitParams):
        # 重新设置训练参数
        self.reset_training_progress_and_params(ts_params.ssnmf_training_params)

        # 将主题ti1拆分为ti1和ti2
        ti1 = ts_params.new_topic1_id
        ti2 = ts_params.new_topic2_id
        new_topic_column_init = self.ss_nmf.w[:, ti1]

        # w为ti2新插入一列，初始化与ti1一样
        self.ss_nmf.w = np.insert(self.ss_nmf.w, ti2, values=new_topic_column_init, axis=1)
        # wr的ti1和ti2列都初始化为原ti1
        self.ss_nmf.wr[:] = 0
        self.ss_nmf.wr[:, ti1] = new_topic_column_init
        self.ss_nmf.wr = np.insert(self.ss_nmf.wr, ti2, values=new_topic_column_init, axis=1)

        # h和hr为ti2新插入一行
        self.ss_nmf.h = np.insert(self.ss_nmf.h, ti2, values=np.zeros(self.ss_nmf.n), axis=0)
        self.ss_nmf.hr[:] = 0
        self.ss_nmf.hr = np.insert(self.ss_nmf.hr, ti2, values=np.zeros(self.ss_nmf.n), axis=0)

        # 为有变动的单词设置新的权重
        for item in ts_params.word_new_weight_dict1.items():
            word_id, new_weight = item
            self.ss_nmf.wr[word_id, ti1] = new_weight

        for item in ts_params.word_new_weight_dict2.items():
            word_id, new_weight = item
            self.ss_nmf.wr[word_id, ti2] = new_weight

        # 将主题数+1
        new_topic_num = self.ss_nmf.k + 1
        self.ss_nmf.k = new_topic_num
        # 重新分配ik，mw矩阵
        self.ss_nmf.ik = np.identity(new_topic_num, dtype=np.int)
        self.ss_nmf.mw = np.zeros((new_topic_num, new_topic_num), dtype=np.float32)
        # 设置监督强度
        self.ss_nmf.mw[ti1, ti1] = ts_params.supervision_intensity1
        self.ss_nmf.mw[ti2, ti2] = ts_params.supervision_intensity2
        # 重制mh
        dn = self.ss_nmf.n
        self.ss_nmf.mh = sparse.csr_matrix((dn, dn), dtype=np.float32)

        # 开始训练
        self.start_nmf_and_umap(is_first_training=False)

    def topic_merge(self, tm_params: TopicMergeParams):
        """
        将topic1_id和topic2_id合并到新主题topic1_id
        topic1_id < topic2_id
        """
        # 重新设置训练参数
        self.reset_training_progress_and_params(tm_params.ssnmf_training_params)

        # 设置主题合并的半监督nmf相关参数、矩阵
        ti1 = tm_params.topic1_id
        ti2 = tm_params.topic2_id

        # 删除h和hr的ti2行，并将h的ti1行（即新主题行）设置为2行的相加
        self.ss_nmf.h[ti1, :] += self.ss_nmf.h[ti2, :]
        self.ss_nmf.h = np.delete(self.ss_nmf.h, ti2, axis=0)
        self.ss_nmf.hr[:] = 0
        self.ss_nmf.hr = np.delete(self.ss_nmf.hr, ti2, axis=0)

        # 删除w和wr的ti2列，并将wr的ti1列（即新主题列）设置为2列的平均
        self.ss_nmf.wr[:] = 0
        self.ss_nmf.wr[:, ti1] = \
            (self.ss_nmf.w[:, ti1] + self.ss_nmf.w[:, ti2]) / 2
        self.ss_nmf.wr = np.delete(self.ss_nmf.wr, ti2, axis=1)
        self.ss_nmf.w = np.delete(self.ss_nmf.w, ti2, axis=1)

        # 为有变动的单词设置新的权重
        for item in tm_params.word_new_weight_dict.items():
            word_id, new_weight = item
            self.ss_nmf.wr[word_id, ti1] = new_weight

        # 主题数-1
        new_topic_num = self.ss_nmf.k - 1
        self.ss_nmf.k = new_topic_num
        # 重新分配ik，mw矩阵
        self.ss_nmf.ik = np.identity(new_topic_num, dtype=np.int)
        self.ss_nmf.mw = np.zeros((new_topic_num, new_topic_num), dtype=np.float32)
        # 设置监督强度
        self.ss_nmf.mw[ti1, ti1] = tm_params.supervision_intensity
        # 重制mh
        dn = self.ss_nmf.n
        self.ss_nmf.mh = sparse.csr_matrix((dn, dn), dtype=np.float32)

        # 开始训练
        self.start_nmf_and_umap(is_first_training=False)

    def keyword_induced_topic_create(self, kitc_params: KeywordInducedTopicCreateParams):
        """
        关键词诱导创建新主题，新主题放在末尾
        """
        # 重新设置训练参数
        self.reset_training_progress_and_params(kitc_params.ssnmf_training_params)

        wn = self.ss_nmf.m
        tn = self.ss_nmf.k
        dn = self.ss_nmf.n

        # w和wr的末尾新插入一列，初始化为0向量
        self.ss_nmf.w = np.insert(self.ss_nmf.w, tn, values=np.zeros(wn), axis=1)
        self.ss_nmf.wr[:] = 0
        self.ss_nmf.wr = np.insert(self.ss_nmf.wr, tn, values=np.zeros(wn), axis=1)
        # 为诱导关键词设置相应的权重
        for item in kitc_params.induce_word_weight_dict.items():
            word_id, new_weight = item
            self.ss_nmf.wr[word_id, tn] = new_weight

        # h和hr的末尾新插入一行，初始化为0向量
        self.ss_nmf.h = np.insert(self.ss_nmf.h, tn, values=np.zeros(dn), axis=0)
        self.ss_nmf.hr[:] = 0
        self.ss_nmf.hr = np.insert(self.ss_nmf.hr, tn, values=np.zeros(dn), axis=0)

        # 主题数+1
        new_topic_num = tn + 1
        self.ss_nmf.k = new_topic_num
        # 重新分配ik，mw矩阵
        self.ss_nmf.ik = np.identity(new_topic_num, dtype=np.int)
        self.ss_nmf.mw = np.zeros((new_topic_num, new_topic_num), dtype=np.float32)
        # 设置监督强度
        self.ss_nmf.mw[tn, tn] = kitc_params.supervision_intensity
        # 重制mh
        self.ss_nmf.mh = sparse.csr_matrix((dn, dn), dtype=np.float32)

        # 开始训练
        self.start_nmf_and_umap(is_first_training=False)

    def document_induced_topic_create(self, ditc_params: DocumentInducedTopicCreateParams):
        """
        文档诱导创建新主题，新主题放在末尾
        """
        # 重新设置训练参数
        self.reset_training_progress_and_params(ditc_params.ssnmf_training_params)

        wn = self.ss_nmf.m
        tn = self.ss_nmf.k
        dn = self.ss_nmf.n

        # h和hr的末尾新插入一行，初始化为0向量
        self.ss_nmf.h = np.insert(self.ss_nmf.h, tn, values=np.zeros(dn), axis=0)
        self.ss_nmf.hr[:] = 0
        self.ss_nmf.hr = np.insert(self.ss_nmf.hr, tn, values=np.zeros(dn), axis=0)
        # 为新主题设置相应的诱导文档相关度
        for item in ditc_params.induce_doc_relevancy_dict.items():
            doc_id, new_relevancy = item
            self.ss_nmf.hr[tn, doc_id] = new_relevancy

        # w和wr的末尾新插入一列，初始化为0向量
        self.ss_nmf.w = np.insert(self.ss_nmf.w, tn, values=np.zeros(wn), axis=1)
        self.ss_nmf.wr[:] = 0
        self.ss_nmf.wr = np.insert(self.ss_nmf.wr, tn, values=np.zeros(wn), axis=1)

        # 主题数+1
        new_topic_num = tn + 1
        self.ss_nmf.k = new_topic_num
        # 重新分配ik，mw矩阵
        self.ss_nmf.ik = np.identity(new_topic_num, dtype=np.int)
        self.ss_nmf.mw = np.zeros((new_topic_num, new_topic_num), dtype=np.float32)
        # 重设mh
        mh_lil = sparse.lil_matrix((dn, dn), dtype=np.float32)
        # 为每个诱导文档列设置相应的监督强度
        for item in ditc_params.induce_doc_supervision_dict.items():
            doc_id, supervision_intensity = item
            mh_lil[doc_id, doc_id] = supervision_intensity
        self.ss_nmf.mh = mh_lil.tocsr()

        # 开始训练
        self.start_nmf_and_umap(is_first_training=False)

    def new_doc_topic_distribution_predict(self, doc):
        """
        新文档主题分布预测
        """
        self.topic_distribution_predict_progress = NewDocTopicDistributionPredictProgress()
        self.topic_distribution_predict_progress.status_code = 1
        # 获取新文档的tf-idf矩阵
        new_doc_tfidf = self.text_preprocessing.get_new_doc_tfidf(doc)
        # 获取新文档的主题分布（即h）
        topic_distribution = self.ss_nmf.fit_new_doc(new_doc_tfidf)
        # 构造主题摘要list，计算每个主题相关度的百分比
        topic_relevancy_sum = np.sum(topic_distribution)
        topic_num = self.ss_nmf.k
        for ti in range(topic_num):
            self.topic_distribution_predict_progress.topic_summary_list.append(
                f"主题{ti}：{self.topic_summary_list[ti]}"
            )
            self.topic_distribution_predict_progress.topic_relevancy_percent_list.append(
                round(topic_distribution[ti, 0] / topic_relevancy_sum * 100, 2)
            )
        # 完成
        self.topic_distribution_predict_progress.status_code = 2

    def get_topic_cluster_data(self, doc_coordinate):
        """
        获取前端主题聚类所需的数据
        """
        is_dimension_3d = self.umap.n_components == 3
        topic_num = self.ss_nmf.k
        docs = self.text_preprocessing.docs

        # 获取每个主题聚类的相关信息
        topic_clusters = []
        for ti in range(topic_num):
            topic_cluster = TopicClusterData()
            # 主题id
            topic_cluster.topic_id = ti
            # 获取主题摘要
            topic_cluster.topic_summary = self.topic_summary_list[ti]
            # 获取该主题聚类下每个文档的相关信息
            topic_cluster.documents = []
            topic_cluster_doc_indexes = self.topic_cluster_doc_indexes_list[ti]
            for di in topic_cluster_doc_indexes:
                document_node = DocumentNodeData()
                # 文档id
                document_node.doc_id = di
                # 文档节选
                doc = docs[di]
                document_node.excerpt = self.text_preprocessing.docExcerptList[di]
                # 文档的UMAP降维2D/3D坐标
                document_node.x = doc_coordinate[di][0]
                document_node.y = doc_coordinate[di][1]
                if is_dimension_3d:
                    document_node.z = doc_coordinate[di][2]
                # 添加到主题文档list
                topic_cluster.documents.append(document_node)

            # 添加到主题集群list
            topic_clusters.append(topic_cluster)

        return topic_clusters

    def get_document_details(self, doc_id):
        document_details = DocumentDetails()
        document_details.doc_id = doc_id
        document_details.text = self.text_preprocessing.docs[doc_id]

        # 获取词频从大到小的单词列表以及词频列表
        bag_words = self.text_preprocessing.bagWords
        word_num = len(bag_words)
        word_frequency_list = self.text_preprocessing.tf[doc_id, :].toarray().reshape(word_num)
        frequency_desc_indexes = np.argsort(word_frequency_list)
        for i in range(word_num):
            word_index = frequency_desc_indexes[i]
            word_frequency = word_frequency_list[word_index]
            if word_frequency > 0:
                document_details.word_frequency_list.append(
                    word_frequency
                )
                document_details.word_list.append(
                    bag_words[word_index]
                )

        # 获取主题相关度从大到小的主题相关度列表以及主题摘要列表
        topic_relevancy_list = self.ss_nmf.h[:, doc_id]
        relevancy_desc_indexes = np.argsort(topic_relevancy_list)
        topic_num = self.ss_nmf.k
        for i in range(topic_num):
            topic_index = relevancy_desc_indexes[i]
            document_details.topic_relevancy_list.append(
                topic_relevancy_list[topic_index]
            )
            document_details.topic_id_list.append(
                topic_index
            )
            document_details.topic_list.append(
                f"主题{topic_index}"
            )
            document_details.topic_summary_list.append(
                self.topic_summary_list[topic_index]
            )

        return document_details

    def get_topic_details(self, topic_id):
        topic_details = TopicDetails()
        topic_details.topic_id = topic_id

        # 获得权重从小到大的主题词项分布（最多100个）
        bag_words = self.text_preprocessing.bagWords
        word_num = len(bag_words)
        max_show_word_num = 100 if word_num > 100 else word_num
        word_distribution_list = self.ss_nmf.w[:, topic_id]
        word_asc_indexes = np.argsort(word_distribution_list)[-max_show_word_num:]
        for i in range(max_show_word_num):
            word_index = word_asc_indexes[i]
            word_distribution = word_distribution_list[word_index]
            topic_details.word_id_list.append(word_index)
            topic_details.word_list.append(bag_words[word_index])
            topic_details.word_distribution_list.append(word_distribution_list[word_index])

        # 获得相关度从小到大的硬聚类到该主题的文档（最多100个）
        topic_cluster_doc_indexes = self.topic_cluster_doc_indexes_list[topic_id]
        if len(topic_cluster_doc_indexes) > 100:
            topic_cluster_doc_indexes = topic_cluster_doc_indexes[-100:]
        topic_relevancy_to_each_doc = self.ss_nmf.h[topic_id, :]
        docs = self.text_preprocessing.docs
        for di in topic_cluster_doc_indexes:
            topic_details.hard_clustering_doc_id_list.append(di)
            topic_details.hard_clustering_doc_list.append(f"文档{di}")
            topic_details.hard_clustering_doc_relevancy_list.append(topic_relevancy_to_each_doc[di])
            topic_details.hard_clustering_doc_text_list.append(docs[di])

        return topic_details

    def get_topic_keyword_info(self, topic_id):
        bag_words = self.text_preprocessing.bagWords
        word_num = len(bag_words)

        keyword_info_list = []
        word_weight_column = self.ss_nmf.w[:, topic_id]
        word_weight_sum = np.sum(word_weight_column)

        for i in range(word_num):
            keyword_info = KeywordInfo()

            keyword_info.word_id = i
            keyword_info.word = bag_words[i]
            keyword_info.word_weight = word_weight_column[i]
            keyword_info.word_weight_percent = \
                round(keyword_info.word_weight / word_weight_sum * 100, 2)

            keyword_info_list.append(keyword_info)

        info = TopicKeywordInfo()
        info.topic_id = topic_id
        info.word_weight_sum = word_weight_sum
        info.keyword_info_list = keyword_info_list

        return info

    def get_topic_merge_keyword_info(self, topic1_id, topic2_id):
        # 始终让topic1_id < topic2_id
        if topic1_id > topic2_id:
            temp = topic1_id
            topic1_id = topic2_id
            topic2_id = temp

        bag_words = self.text_preprocessing.bagWords
        word_num = len(bag_words)

        keyword_info_list = []
        word_weight_column1 = self.ss_nmf.w[:, topic1_id]
        word_weight_column2 = self.ss_nmf.w[:, topic2_id]
        word_weight_sum = (np.sum(word_weight_column1) + np.sum(word_weight_column2)) / 2

        for i in range(word_num):
            keyword_info = KeywordInfo()

            keyword_info.word_id = i
            keyword_info.word = bag_words[i]
            keyword_info.word_weight = (word_weight_column1[i] + word_weight_column2[i]) / 2
            keyword_info.word_weight_percent = \
                round(keyword_info.word_weight / word_weight_sum * 100, 2)

            keyword_info_list.append(keyword_info)

        info = TopicMergeKeywordInfo()
        info.topic1_id = topic1_id
        info.topic2_id = topic2_id
        info.word_weight_sum = word_weight_sum
        info.keyword_info_list = keyword_info_list

        return info
