import json
from models import ChineseLyric
from database import SessionLocal

music_json = None
try:
    with open("resource/corpus/music.json", 'r', encoding='UTF-8') as music_file:
        music_json = json.load(music_file)
except IOError as e:
    print('文件读取异常！', e.msg)
else:
    song_list = []
    for song in music_json:
        # 歌手
        singer = song["singer"]
        # 歌名
        song_name = song["song"]
        # 专辑名
        album = song["album"]
        # 歌词
        lyric = '\n'.join(song["geci"])
        lyric = f"{song_name}\n\t{singer} - {album}\n{lyric}"
        song_list.append(ChineseLyric(
            singer=singer,
            song=song_name,
            album=album,
            text=lyric
        ))

    # 写入数据库
    try:
        session = SessionLocal()
        session.add_all(song_list)
    except Exception as e:
        print(e)
        session.close()
    else:
        session.commit()


