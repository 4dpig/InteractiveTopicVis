from sqlalchemy.orm import Session
import models
import schemas


def get_bilibili_videos(db: Session):
    try:
        return db.query(models.BilibiliVideo).all()
    except Exception as e:
         return None


def get_bilibili_videos_by_vid(vid, db: Session):
    try:
        return db.query(models.BilibiliVideo).\
            filter(models.BilibiliVideo.id == vid).one()
    except Exception as e:
        return None


def get_bilibili_video_comments_by_vid(vid, db: Session):
    try:
        return db.query(models.BilibiliVideoComment). \
            filter(models.BilibiliVideoComment.vid == vid).all()
    except Exception as e:
        return None


def get_online_shopping_reviews(db: Session):
    try:
        return db.query(models.OnlineShoppingReview).all()
    except Exception as e:
        return None


def get_takeaway_reviews(db: Session):
    try:
        return db.query(models.TakeawayReview).all()
    except Exception as e:
        return None


def get_chinese_lyrics(db: Session):
    try:
        return db.query(models.ChineseLyric).all()
    except Exception as e:
        return None


def get_COVID19_news(db: Session):
    try:
        return db.query(models.COVID19News).all()
    except Exception as e:
        return None
