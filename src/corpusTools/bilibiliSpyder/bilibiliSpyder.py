import re
import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout, MissingSchema
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import streamlit as st
import bilibiliJsonTool


@st.cache(show_spinner=False)
def send_request(request_url, params={}, timeout=3):
    """
    发送HTTP请求
    """
    try:

        r = requests.get(request_url, params=params, timeout=timeout,
                         headers={'User-Agent': 'Chrome'})

    except (ConnectionError, HTTPError, Timeout, MissingSchema) as e:
        return None
    else:
        if r.status_code == requests.codes.ok:
            return r
        else:
            return None


@st.cache
def get_oid_by_bvid(bvid):
    response = send_request(
        f'https://api.bilibili.com/x/web-interface/archive/stat',
        params={'bvid': bvid}
    )

    if not response:
        return ''
    else:
        return bilibiliJsonTool.get_oid_by_bvid_response(response.text)


@st.cache
def get_video_info_by_url(video_url):
    """
    通过视频url获取视频信息，例：
    {
        'oid': '82075461',
        'type': '电影',
        'title': '功夫'
    }

    由于是动态网页，所以需要用Selenium + Chrome Headless
    """
    # 创建chrome启动选项
    chrome_options = webdriver.ChromeOptions()
    # 指定chrome启动类型为headless
    chrome_options.add_argument('--headless')

    # 调用环境变量指定的chromedriver位置来创建浏览器对象如：
    # C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe
    # 如果没有添加环境变量，可通过传递 executable_path 参数来显式指定chromedriver的位置
    # chromedriver不是安装了Chrome就有，需要额外下载
    driver = webdriver.Chrome(chrome_options=chrome_options)

    # 等待页面加载完成
    video_info = {}
    driver.get(video_url)
    try:
        WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "a[class='av-link'], div[id='viewbox_report']")
                )
        )

    except TimeoutException as e:
        print('页面加载超时！', e.msg)
    else:
        # 获取网页源码字符串
        page_source = driver.page_source
        # 解析
        bs = BeautifulSoup(page_source, 'html.parser')

        # 查看url中是否包含bvid（即BVXXX，b站视频url有好几种，有epxxx、ssxxx等等，点开之后还会重定向）
        url_type_pattern = re.compile(r'(?<=BV)([a-zA-Z0-9]+)')
        url_match = url_type_pattern.search(video_url)

        # 获取bvid
        bvid = ''
        if url_match:
            bvid = url_match.group(0)
        else:
            # 获取oid
            bvid_a = bs.find('a', attrs={'class': 'av-link'}, text=re.compile(r'BV[a-zA-Z0-9]+'))
            if bvid_a:
                bvid = re.compile(r'(?<=BV)([a-zA-Z0-9]+)').search(bvid_a.string).group(0)

        # 根据bvid获取oid
        oid = get_oid_by_bvid(bvid)
        if oid:
            video_info['oid'] = oid

        # 获取分区（多级）
        a_crumbs = bs.find('span', attrs={'class' : 'a-crumbs'})
        if a_crumbs:
            a_list = a_crumbs.find_all('a')
            if a_list:
                video_type_list = [a.string for a in a_list]
                video_type_format = '{}>' * (len(video_type_list) - 1) + '{}'
                video_info['type'] = video_type_format.format(*video_type_list)
        else:
            # 获取分区（只可能是单级）
            type_a = bs.find('a', attrs={'class': 'home-link'})
            if type_a:
                video_info['type'] = type_a.string

        # 获取视频标题
        title_h1 = bs.find('h1', attrs={'class': 'video-title'})
        if title_h1:
            video_info['title'] = title_h1['title']
        else:
            title_a = bs.find('a', attrs={'class': 'media-title'})
            if title_a:
                video_info['title'] = title_a.string
    finally:
        # 关闭浏览器
        driver.quit()
        return video_info


@st.cache(suppress_st_warning=True)
def get_comments_json_by_oid(oid, max_page_number = 1, show_progress=False):
    """
    b站视频评论API：
    https://api.bilibili.com/x/v2/reply?
        jsonp=jsonp& # 固定
        pn=3& # 评论页码
        type=1& # 一般固定
        oid={视频id}&
        sort=2 # 评论排序方式，2应该是按热度排序，一般固定
    """

    progress_text = None
    if show_progress:
        progress_text = st.empty()

    # 构造url和参数
    url = "https://api.bilibili.com/x/v2/reply"
    params = {
        'jsonp': 'jsonp',
        'pn': 1,
        'type': 1,
        'oid': oid,
        'sort': 2
    }

    # 获取每个页码的评论json数据
    comments_json_arary = []
    for pageNumber in range(1, max_page_number + 1):
        if show_progress:
            progress_text.text(f"正在抓取第{pageNumber}页")

        params['pn'] = pageNumber
        response = send_request(url, params)
        if response:
            comments_json_arary.append(response.text)

    return comments_json_arary


