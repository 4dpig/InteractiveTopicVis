import json
import streamlit as st
from database import SessionLocal
from models import BilibiliVideo, BilibiliVideoComment
import sqlalchemy
import re


def is_video_data_exists(oid):
    """
    查询bilibiliVideos表中是否已存在该视频
    """
    try:
        session = SessionLocal()
        # 查询是否存在
        count = session.query(BilibiliVideo).filter(BilibiliVideo.oid == oid).count()
    except Exception as e:
        print(e)
        session.close()
        return False
    else:
        session.commit()
        return count > 0


def delete_video_by_oid(oid):
    """
    查询bilibiliVideos表中是否已存在该视频
    """
    try:
        session = SessionLocal()
        # 查询是否存在
        del_video = session.query(BilibiliVideo).filter(BilibiliVideo.oid == oid).first()
        session.delete(del_video)
    except Exception as e:
        print(e)
        session.close()
        return False
    else:
        session.commit()
        return True


def write_video_comments_sql(isVideoExists, video_info, comments_json_array):
    """
    将评论数据写入数据库
    """
    if isVideoExists:
        result = delete_video_by_oid(video_info['oid'])
        if not result:
            return False

    try:
        session = SessionLocal()

        # 写入bilibiliVideos
        video_record = BilibiliVideo(
            oid=video_info['oid'],
            url=video_info['url'],
            type=video_info['type'],
            title=video_info['title']
        )

        session.add(video_record)
        session.flush()

        # 写入bilibiliVideoComments
        vid = video_record.id
        commentRecords = []

        for page_json in comments_json_array:
            page_comment = json.loads(page_json)
            top_level_replies = page_comment['data']['replies']
            for reply in top_level_replies:
                # 去除表情符号
                text = re.sub(r'\[\S+\]', '', reply['content']['message'])
                comment = BilibiliVideoComment(
                    vid=vid,
                    text=text
                )
                commentRecords.append(comment)

        session.add_all(commentRecords)

    except Exception as e:
        print(e)
        session.close()
        return False
    else:
        session.commit()
        return True


