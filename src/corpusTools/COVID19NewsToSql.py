import pandas as pd
from models import COVID19News
from database import SessionLocal
from datetime import datetime

df = pd.read_csv("resource/corpus/news.csv")

news_list = []
for row in df.itertuples():
    title = getattr(row, "title")
    publish_date_str = getattr(row, "publish_date")
    publish_date = datetime.strptime(publish_date_str, "%Y-%m-%d %H:%M:%S")
    text = getattr(row, "text")
    news = COVID19News(
        title=title,
        publish_date=publish_date,
        text=text
    )

    news_list.append(news)

# 写入数据库
try:
    session = SessionLocal()
    session.add_all(news_list)
except Exception as e:
    print(e)
    session.close()
else:
    session.commit()


