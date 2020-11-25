import json
import os
import streamlit as st
import re


@st.cache
def get_oid_by_bvid_response(bvid_response):
    json_data = json.loads(bvid_response, encoding='UTF-8')
    return str(json_data['data']['aid'])


@st.cache
def video_comments_data_exists(oid):
    """
    检查视频的评论json数据是否已抓取
    文件名就是{oid}.json
    """
    return os.path.isfile('resource/corpus/bilibiliVideoComments/{}.json'.format(oid))


@st.cache
def get_page_count(comments_json_text):
    """
    计算视频评论总页数
    """
    try:
        comments_json = json.loads(comments_json_text)
        data = comments_json['data']
        page = data['page']
        comments_each_page = page['size']
        total_comments_number = page['count']
        return int(total_comments_number / comments_each_page) + 1
    except json.JSONDecodeError as e:
        print("不是有效的JSON格式！", e.msg)
        return -1


@st.cache
def get_comments_json_path_by_oid(oid):
    return os.path.join(
        'resource/corpus/bilibiliVideoComments', oid + '.json')


@st.cache
def read_video_comments_json_by_oid(oid):
    """
    根据oid读取相应的json文件
    """
    comments_json_path = get_comments_json_path_by_oid(oid)

    try:
        with open(comments_json_path, 'r', encoding='UTF-8') as comments_file:
            return json.load(comments_file)
    except IOError as e:
        print('文件读取异常！', e.msg)
        return {}


def read_all_video_comments_json():
    """
    阅读所有的评论json文件,返回
    {
        oid: comments_json,
        ...
    }
    形式的字典
    """
    comments_json_dir = 'resource/corpus/bilibiliVideoComments'

    comments_json_dict = {}
    with os.scandir(comments_json_dir) as it:
        for entry in it:
            if entry.name.endswith('.json'):
                try:
                    with open(entry.path, 'r', encoding='UTF-8') as comments_file:
                        oid, _ = os.path.splitext(entry.name)
                        comments_json_dict[oid] = json.load(comments_file)
                except IOError as e:
                    print(entry.name, '文件读取异常！', e.msg)

    return comments_json_dict


@st.cache
def write_video_comments_json(video_info, comments_json_array):
    """
    读取返回的评论json数据，
    提取想要的数据，并写入json文件
    """
    # 查找所有的顶级评论（不包括评论的回复、楼中楼）
    comments = []

    for page_json in comments_json_array:
        page_comment = json.loads(page_json)
        top_level_replies = page_comment['data']['replies']
        for reply in top_level_replies:
            # 去除表情符号
            comment = re.sub(r'\[\S+\]', '', reply['content']['message'])
            comments.append(comment)

    # 文件名
    comments_json_path = get_comments_json_path_by_oid(video_info['oid'])

    # 构造json格式
    comments_json = video_info.copy()
    comments_json['comments'] = comments

    # 写入文件
    try:
        with open(comments_json_path, 'w', encoding='UTF-8') as comments_file:
            json.dump(comments_json, comments_file, ensure_ascii=False)
    except IOError as e:
        print('文件写入异常！', e.msg)
        return False
    else:
        return True
