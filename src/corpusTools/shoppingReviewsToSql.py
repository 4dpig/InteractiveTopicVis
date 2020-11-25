import pandas as pd
from models import OnlineShoppingReview
from database import SessionLocal

df = pd.read_csv("resource/corpus/online_shopping_10_cats.csv")

review_list = []
for row in df.itertuples():
    item_type = getattr(row, "cat")
    review_type = getattr(row, "label")
    review_type = "好评" if review_type == 1 else "差评"
    text = getattr(row, "review")
    if not pd.isnull(text):
        review = OnlineShoppingReview(
            item_type=item_type,
            review_type=review_type,
            text=text
        )

        review_list.append(review)

# 写入数据库
try:
    session = SessionLocal()
    session.add_all(review_list)
except Exception as e:
    print(e)
    session.close()
else:
    session.commit()


