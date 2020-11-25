from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./resource/sqlite/corpus.db"

# 数据库连接
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 数据库会话类
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 数据库表映射类的基类
Base = declarative_base()
