# 使用官方的 Python 3.12 slim 版本作為基礎映像檔
FROM python:3.12-slim

# 設定環境變數，讓 Python 的輸出直接顯示，方便在 Cloud Run 中查看日誌
ENV PYTHONUNBUFFERED True

# 設定容器內的工作目錄
WORKDIR /app

# 先複製 requirements.txt 並安裝套件
# 這樣做可以利用 Docker 的層快取，如果套件沒有變更，就不用重新安裝
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 將專案的所有檔案複製到工作目錄中
COPY . .

# 設定容器啟動時要執行的指令
# 使用 gunicorn 來啟動 Flask 應用程式 (app:app 指的是 app.py 檔案中的 app 物件)
# --bind :$PORT 會讓 gunicorn 監聽 Cloud Run 自動提供的連接埠
# --workers, --threads 是效能調校參數，可以根據您的需求調整
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
