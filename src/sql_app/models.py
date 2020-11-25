# 数据库模型
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from database import Base


# bilibili视频映射类
class BilibiliVideo(Base):
    __tablename__ = 'bilibiliVideos'

    id = Column(Integer, primary_key=True, index=True)
    oid = Column(String, unique=True, nullable=False, index=True)
    url = Column(String, unique=True, nullable=False, index=True)
    type = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    comments = relationship("BilibiliVideoComment", back_populates='video',
                            cascade="all, delete, delete-orphan")


# bilibili视频评论映射类
class BilibiliVideoComment(Base):
    __tablename__ = 'bilibiliVideoComments'

    id = Column(Integer, primary_key=True, index=True)
    vid = Column(Integer, ForeignKey('bilibiliVideos.id'))
    text = Column(String, nullable=False)
    video = relationship("BilibiliVideo", back_populates="comments")


# 外卖评价语料映射类
class TakeawayReview(Base):
    __tablename__ = 'takeawayReviews'

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, index=True)
    text = Column(String, nullable=False)


# 电商购物评价语料映射类
class OnlineShoppingReview(Base):
    __tablename__ = 'onlineShoppingReviews'

    id = Column(Integer, primary_key=True, index=True)
    item_type = Column(String, index=True)
    review_type = Column(String, index=True)
    text = Column(String, nullable=False)


# 中文歌歌词语料映射类
class ChineseLyric(Base):
    __tablename__ = 'chineseLyrics'

    id = Column(Integer, primary_key=True, index=True)
    singer = Column(String, index=True)
    song = Column(String, index=True)
    album = Column(String, index=True)
    text = Column(String, nullable=False)


# 新冠病毒新闻语料映射类
class COVID19News(Base):
    __tablename__ = 'COVID19News'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    publish_date = Column(DateTime, index=True)
    text = Column(String, nullable=False)


# 在数据库中根据映射类创建相应的表
# from database import engine
# Base.metadata.create_all(engine)
