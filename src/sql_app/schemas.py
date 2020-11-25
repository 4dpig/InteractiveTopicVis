# Pydantic模型，用于前后端数据转换 json<->类
from pydantic import BaseModel
from datetime import datetime
from typing import List


# bilibili视频评论模型类
class BilibiliVideoComment(BaseModel):
    id: int
    vid: int
    text: str

    class Config:
        orm_mode = True


# bilibili视频模型类
class BilibiliVideo(BaseModel):
    id: int
    oid: str
    url: str
    type: str
    title: str

    class Config:
        orm_mode = True


# 外卖评价语料映射类
class TakeawayReview(BaseModel):
    id: int
    type: str
    text: str

    class Config:
        orm_mode = True


# 电商购物评价语料映射类
class OnlineShoppingReview(BaseModel):
    id: int
    item_type: str
    review_type: str
    text: str

    class Config:
        orm_mode = True


# 中文歌歌词语料映射类
class ChineseLyrics(BaseModel):
    id: int
    singer: str
    song: str
    album: str
    text: str

    class Config:
        orm_mode = True


# 新冠病毒新闻语料映射类
class COVID19News(BaseModel):
    id: int
    title: str
    publish_date: datetime
    text: str

    class Config:
        orm_mode = True
