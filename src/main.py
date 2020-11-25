from typing import List

from fastapi import Depends, FastAPI, BackgroundTasks, Response, Cookie, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional, Dict
import crud, models, schemas
from database import SessionLocal
from TopicModelTrainingTask import *
import uuid

app = FastAPI()

# 配置跨域问题
origins = [
    "http://localhost:8000",
    "http://localhost:3000"
]

# 全局主题模型训练对象
topic_model_training_tasks: Dict[str, TopicModelTrainingTask] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据库连接依赖
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@app.get(
    "/bilibilivideos/",
    response_model=List[schemas.BilibiliVideo],
    tags=['bilibili视频评论语料库数据接口']
)
def get_bilibili_videos(db: Session = Depends(get_db)):
    db_videos = crud.get_bilibili_videos(db)
    return db_videos


@app.get(
    "/bilibilivideos/{vid}/",
    response_model=schemas.BilibiliVideo,
    tags=['bilibili视频评论语料库数据接口']
)
def get_bilibili_videos_by_vid(vid: int, db: Session = Depends(get_db)):
    db_video = crud.get_bilibili_videos_by_vid(vid, db)
    return db_video


@app.get(
    "/bilibilivideos/{vid}/comments/",
    response_model=List[schemas.BilibiliVideoComment],
    tags=['bilibili视频评论语料库数据接口']
)
def get_bilibili_video_comments_by_vid(vid: int, db: Session = Depends(get_db)):
    db_comments = crud.get_bilibili_video_comments_by_vid(vid, db)
    return db_comments


@app.get(
    "/onlineShoppingReviews/",
    response_model=List[schemas.OnlineShoppingReview],
    tags=['电商购物评价语料库数据接口']
)
def get_online_shopping_reviews(db: Session = Depends(get_db)):
    db_reviews = crud.get_online_shopping_reviews(db)
    return db_reviews


@app.get(
    "/takeawayReviews/",
    response_model=List[schemas.TakeawayReview],
    tags=['外卖评价语料库数据接口']
)
def get_takeaway_reviews(db: Session = Depends(get_db)):
    db_reviews = crud.get_takeaway_reviews(db)
    return db_reviews


@app.get(
    "/chineseLyrics/",
    response_model=List[schemas.ChineseLyrics],
    tags=['中文歌歌词语料库数据接口']
)
def get_chinese_lyrics(db: Session = Depends(get_db)):
    db_reviews = crud.get_chinese_lyrics(db)
    return db_reviews


@app.get(
    "/COVID19News/",
    response_model=List[schemas.COVID19News],
    tags=['新冠病毒新闻语料库数据接口']
)
def get_COVID19_news(db: Session = Depends(get_db)):
    db_reviews = crud.get_COVID19_news(db)
    return db_reviews


# 为每个用户设置uuid标识符，并存入Cookie
def create_user_uuid_and_set_cookie(response: Response):
    user_uuid = str(uuid.uuid1())
    response.set_cookie(key="user_uuid", value=user_uuid)
    return user_uuid


@app.post(
    "/calculate/preprocessing/",
    tags=["计算模块数据接口"]
)
async def run_text_preprocessing_task(text_preprocessing_params: TextPreprocessingParams,
                                      background_task: BackgroundTasks,
                                      response: Response,
                                      user_uuid: str = Cookie(None)):
    if not user_uuid:
        user_uuid = create_user_uuid_and_set_cookie(response)

    global topic_model_training_tasks
    task = TopicModelTrainingTask(text_preprocessing_params)
    topic_model_training_tasks[user_uuid] = task

    background_task.add_task(task.preprocessing)
    return {"message": "已添加文本预处理后台任务"}


@app.post(
    "/calculate/nmftraining/",
    tags=["计算模块数据接口"]
)
async def run_nmf_training_and_tsne_task(nmf_training_params: NMFTrainingParams,
                                         background_task: BackgroundTasks,
                                         response: Response,
                                         user_uuid: str = Cookie(None)):
    if not user_uuid:
        user_uuid = create_user_uuid_and_set_cookie(response)

    global topic_model_training_tasks

    if user_uuid and user_uuid in topic_model_training_tasks:
        task = topic_model_training_tasks[user_uuid]
        if task.text_preprocessing_progress.status_code == 2:
            background_task.add_task(task.nmf_training, nmf_training_params)
            return {"message": "已添加NMF主题模型训练后台任务"}
        else:
            raise HTTPException(status_code=404, detail="请先进行文本预处理任务！")
    else:
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")


@app.get(
    "/calculate/nmftraining/keywordsearch/",
    response_model=KeywordSearchResult,
    tags=["用户交互"]
)
async def search_keyword(search_text: str,
                         response: Response,
                         user_uuid: str = Cookie(None)):
    if not user_uuid:
        user_uuid = create_user_uuid_and_set_cookie(response)

    global topic_model_training_tasks

    if user_uuid and user_uuid in topic_model_training_tasks:
        task = topic_model_training_tasks[user_uuid]
        if task.text_preprocessing_progress.status_code == 2:
            search_text = search_text.lower()
            bag_words = task.text_preprocessing.bagWords
            keyword_search_result = KeywordSearchResult()
            for wi, word in enumerate(bag_words):
                if search_text in word:
                    keyword_search_result.word_id_list.append(wi)
                    keyword_search_result.word_list.append(word)
            return keyword_search_result
        else:
            raise HTTPException(status_code=404, detail="请先进行文本预处理任务！")
    else:
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")


@app.post(
    "/calculate/nmftraining/topickeywordoptimization/",
    tags=["用户交互"]
)
async def run_topic_keyword_optimization_task(tko_params: TopicKeywordOptimizationParams,
                                              background_task: BackgroundTasks,
                                              response: Response,
                                              user_uuid: str = Cookie(None)):
    if not user_uuid:
        user_uuid = create_user_uuid_and_set_cookie(response)

    global topic_model_training_tasks

    if user_uuid and user_uuid in topic_model_training_tasks:
        task = topic_model_training_tasks[user_uuid]
        if task.nmf_training_progress.status_code == 2:
            background_task.add_task(task.topic_keyword_optimization, tko_params)
            return {"message": "已添加主题关键词优化后台任务"}
        else:
            raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")
    else:
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")


@app.post(
    "/calculate/nmftraining/topicsplit/",
    tags=["用户交互"]
)
async def run_topic_split_task(ts_params: TopicSplitParams,
                               background_task: BackgroundTasks,
                               response: Response,
                               user_uuid: str = Cookie(None)):
    if not user_uuid:
        user_uuid = create_user_uuid_and_set_cookie(response)

    global topic_model_training_tasks

    if user_uuid and user_uuid in topic_model_training_tasks:
        task = topic_model_training_tasks[user_uuid]
        if task.nmf_training_progress.status_code == 2:
            background_task.add_task(task.topic_split, ts_params)
            return {"message": "已添加主题拆分后台任务"}
        else:
            raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")
    else:
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")


@app.post(
    "/calculate/nmftraining/topicmerge/",
    tags=["用户交互"]
)
async def run_topic_merge_task(tm_params: TopicMergeParams,
                               background_task: BackgroundTasks,
                               response: Response,
                               user_uuid: str = Cookie(None)):
    if not user_uuid:
        user_uuid = create_user_uuid_and_set_cookie(response)

    global topic_model_training_tasks

    if user_uuid and user_uuid in topic_model_training_tasks:
        task = topic_model_training_tasks[user_uuid]
        if task.nmf_training_progress.status_code == 2:
            background_task.add_task(task.topic_merge, tm_params)
            return {"message": "已添加主题合并后台任务"}
        else:
            raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")
    else:
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")


@app.post(
    "/calculate/nmftraining/keywordinducedtopiccreate/",
    tags=["用户交互"]
)
async def run_keyword_induced_topic_create_task(kitc_params: KeywordInducedTopicCreateParams,
                                                background_task: BackgroundTasks,
                                                response: Response,
                                                user_uuid: str = Cookie(None)):
    if not user_uuid:
        user_uuid = create_user_uuid_and_set_cookie(response)

    global topic_model_training_tasks

    if user_uuid and user_uuid in topic_model_training_tasks:
        task = topic_model_training_tasks[user_uuid]
        if task.nmf_training_progress.status_code == 2:
            background_task.add_task(task.keyword_induced_topic_create, kitc_params)
            return {"message": "已添加关键词诱导主题创建后台任务"}
        else:
            raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")
    else:
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")


@app.post(
    "/calculate/nmftraining/documentinducedtopiccreate/",
    tags=["用户交互"]
)
async def run_document_induced_topic_create_task(ditc_params: DocumentInducedTopicCreateParams,
                                                 background_task: BackgroundTasks,
                                                 response: Response,
                                                 user_uuid: str = Cookie(None)):
    if not user_uuid:
        user_uuid = create_user_uuid_and_set_cookie(response)

    global topic_model_training_tasks

    if user_uuid and user_uuid in topic_model_training_tasks:
        task = topic_model_training_tasks[user_uuid]
        if task.nmf_training_progress.status_code == 2:
            background_task.add_task(task.document_induced_topic_create, ditc_params)
            return {"message": "已添加文档诱导主题创建后台任务"}
        else:
            raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")
    else:
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")


@app.get(
    "/calculate/nmftraining/newdoctopicdistributionpredict/",
    tags=["主题预测"]
)
async def run_new_document_topic_distribution_predict_task(
        new_doc_text: str,
        background_task: BackgroundTasks,
        response: Response,
        user_uuid: str = Cookie(None)):
    if not user_uuid:
        user_uuid = create_user_uuid_and_set_cookie(response)

    global topic_model_training_tasks

    if user_uuid and user_uuid in topic_model_training_tasks:
        task = topic_model_training_tasks[user_uuid]
        if task.nmf_training_progress.status_code == 2:
            background_task.add_task(
                task.new_doc_topic_distribution_predict,
                new_doc_text
            )
            return {"message": "已添加新文档主题分布预测后台任务"}
        else:
            raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")
    else:
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")


@app.get(
    "/calculate/preprocessing/progress/",
    tags=["计算模块数据接口"],
    response_model=TextPreprocessingProgress
)
def get_text_preprocessing_task_progress(user_uuid: str = Cookie(None)):
    global topic_model_training_tasks

    if (not user_uuid) or (user_uuid not in topic_model_training_tasks):
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")
    return topic_model_training_tasks[user_uuid].text_preprocessing_progress


@app.get(
    "/calculate/nmftraining/progress/",
    tags=["计算模块数据接口"],
    response_model=NMFTrainingProgress
)
def get_nmf_training_task_progress(user_uuid: str = Cookie(None)):
    global topic_model_training_tasks

    if (not user_uuid) or (user_uuid not in topic_model_training_tasks):
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")

    task = topic_model_training_tasks[user_uuid]
    if task.text_preprocessing_progress.status_code == 2:
        return task.nmf_training_progress
    else:
        raise HTTPException(status_code=404, detail="请先进行文本预处理任务！")


@app.get(
    "/calculate/umap/progress/",
    tags=["计算模块数据接口"],
    response_model=UMAPProgress
)
def get_umap_task_progress(user_uuid: str = Cookie(None)):
    global topic_model_training_tasks

    if (not user_uuid) or (user_uuid not in topic_model_training_tasks):
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")

    task = topic_model_training_tasks[user_uuid]
    if task.nmf_training_progress.status_code == 2:
        return task.umap_progress
    else:
        raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")


@app.get(
    "/calculate/nmftraining/predict/progress/",
    tags=["计算模块数据接口"],
    response_model=NewDocTopicDistributionPredictProgress
)
def get_new_doc_topic_distribution_predict_task_progress(user_uuid: str = Cookie(None)):
    global topic_model_training_tasks

    if (not user_uuid) or (user_uuid not in topic_model_training_tasks):
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")

    task = topic_model_training_tasks[user_uuid]
    if task.nmf_training_progress.status_code == 2:
        return task.topic_distribution_predict_progress
    else:
        raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")


@app.get(
    "/calculate/details/document/",
    tags=["计算模块数据接口"],
    response_model=DocumentDetails
)
def get_document_details(doc_id: int, user_uuid: str = Cookie(None)):
    global topic_model_training_tasks

    if (not user_uuid) or (user_uuid not in topic_model_training_tasks):
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")

    task = topic_model_training_tasks[user_uuid]
    if task.nmf_training_progress.status_code == 2:
        return task.get_document_details(doc_id)
    else:
        raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")


@app.get(
    "/calculate/details/topic/",
    tags=["计算模块数据接口"],
    response_model=TopicDetails
)
def get_topic_details(topic_id: int, user_uuid: str = Cookie(None)):
    global topic_model_training_tasks

    if (not user_uuid) or (user_uuid not in topic_model_training_tasks):
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")

    task = topic_model_training_tasks[user_uuid]
    if task.nmf_training_progress.status_code == 2:
        return task.get_topic_details(topic_id)
    else:
        raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")


@app.get(
    "/calculate/userinteraction/info/keyword/",
    tags=["计算模块数据接口"],
    response_model=TopicKeywordInfo
)
def get_topic_keyword_info(topic_id: int, user_uuid: str = Cookie(None)):
    global topic_model_training_tasks

    if (not user_uuid) or (user_uuid not in topic_model_training_tasks):
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")

    task = topic_model_training_tasks[user_uuid]
    if task.nmf_training_progress.status_code == 2:
        return task.get_topic_keyword_info(topic_id)
    else:
        raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")


@app.get(
    "/calculate/userinteraction/info/topicmergekeyword/",
    tags=["计算模块数据接口"],
    response_model=TopicMergeKeywordInfo
)
def get_topic_merge_keyword_info(topic1_id: int, topic2_id: int, user_uuid: str = Cookie(None)):
    global topic_model_training_tasks

    if (not user_uuid) or (user_uuid not in topic_model_training_tasks):
        raise HTTPException(status_code=404, detail="未找到相应的训练对象！")

    task = topic_model_training_tasks[user_uuid]
    if task.nmf_training_progress.status_code == 2:
        return task.get_topic_merge_keyword_info(topic1_id, topic2_id)
    else:
        raise HTTPException(status_code=404, detail="请先等待NMF训练结束！")
