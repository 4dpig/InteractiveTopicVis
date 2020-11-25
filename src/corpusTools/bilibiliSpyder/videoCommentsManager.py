import pandas as pd
import streamlit as st
from PIL import Image

import bilibiliJsonTool
import bilibiliSpyder
import bilibiliSqlTool


class VideoCommentsManager:
    """
    视频评论管理界面类
    """

    def __init__(self):
        self.video_info = {}

    def video_info_part(self):
        with st.spinner('获取视频基本信息中...'):
            video_info = bilibiliSpyder.get_video_info_by_url(self.video_info['url'])

        if 'oid' in video_info.keys():
            self.video_info.update(video_info)

            self.isVideoExists = \
                bilibiliSqlTool.is_video_data_exists(self.video_info['oid']) if \
                    self.isWriteSQL else \
                    bilibiliJsonTool.video_comments_data_exists(self.video_info['oid'])

            comments_data_exists = \
                '已抓取(可重新抓取覆盖)' if \
                    self.isVideoExists else \
                    '未抓取'

            # 显示视频基本信息
            pd_video_info = pd.DataFrame({
                '视频编号': [self.video_info['oid']],
                '视频分区': [self.video_info['type']],
                '视频标题': [self.video_info['title']],
                '该视频评论抓取情况': [comments_data_exists]
            })

            # 以表格的方式显示视频信息
            st.table(pd_video_info)

            return True
        else:
            st.error('无法获取视频信息！')
            return False

    def comments_scarping_part(self):
        # 获取评论总页数
        single_page_comments = bilibiliSpyder.get_comments_json_by_oid(self.video_info['oid'])

        if single_page_comments:
            page_count = bilibiliJsonTool.get_page_count(single_page_comments[0])
            # 让用户选择需要爬取的页数
            max_page_number = st.slider('选择需要抓取评论的最大页数：', 1, page_count, page_count)

            if st.button('开始抓取：'):
                # 获取评论数据
                with st.spinner('获取评论数据中...'):
                    comments_json_array = bilibiliSpyder. \
                        get_comments_json_by_oid(
                        self.video_info['oid'],
                        max_page_number,
                        show_progress=True)

                if comments_json_array:
                    # 写入本地
                    if self.isWriteSQL:
                        with st.spinner('写入数据库中...'):
                            writeResult = bilibiliSqlTool. \
                                write_video_comments_sql(self.isVideoExists, self.video_info,
                                                         comments_json_array)
                    else:
                        with st.spinner('写入json文件中...'):
                            writeResult = bilibiliJsonTool. \
                                write_video_comments_json(self.video_info, comments_json_array)

                    if writeResult:
                        return True
                    else:
                        st.error('无法写入数据库...'
                                 if self.isWriteSQL
                                 else '无法写入json文件...'
                                 )
                        return False
                else:
                    st.error('无法获取评论数据！')
                    return False
            else:
                return False
        else:
            st.error('无法获取评论总页数！')
            return False

    def displayScarpingPage(self):
        """
        评论抓取页面主流程函数
        """

        # 写入类型选择
        write_option = st.sidebar.selectbox('爬取数据写入选择：', ('写入数据库', '写入json文件'))
        self.isWriteSQL = write_option == '写入数据库'

        st.title('bilibili视频评论抓取')

        # 视频选择
        st.header('选择视频')
        # 显示图片
        image = Image.open('resource/picture/fast_copy_video_link.png')
        st.image(image, caption='如何快速复制视频URL')

        show_embed_website = st.checkbox('显示/隐藏内嵌网页')
        if show_embed_website:
            # 嵌入网页
            st.markdown("""
            <iframe name='bilibili-embed'
                    src='https://www.bilibili.com/'
                    width='100%' 
                    height='500' 
                    scrolling='yes'
                    frameborder='0'
            </iframe>""", unsafe_allow_html=True)

        # 用户输入url
        url = st.text_input('输入所要抓取评论的视频URL：')

        if url:
            self.video_info['url'] = url

            # 获取视频基本信息
            st.header('视频基本信息：')
            if self.video_info_part():
                st.header('评论抓取：')
                if self.comments_scarping_part():
                    st.success('评论数据抓取完成，已写入本地！')
                    st.balloons()
