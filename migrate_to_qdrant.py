import os
import uuid
import pyodbc
import mysql.connector
import google.generativeai as genai
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# --- 設定 ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

COLLECTION_NAME = "taiwan_food_menu"
VECTOR_SIZE = 768

# --- 通用資料庫連線邏輯 ---

def _connect_sql_server():
    """
    建立並回傳一個 SQL Server 的連線物件。
    自動偵測帳號密碼或 Windows 整合式驗證。
    """
    try:
        driver = os.environ.get('DB_DRIVER', '{ODBC Driver 17 for SQL Server}')
        server = os.environ.get('DB_SERVER', 'localhost')
        database = os.environ.get('DB_DATABASE', 'Menu')
        username = os.environ.get('DB_UID')
        password = os.environ.get('DB_PWD')

        auth_part = ""
        # 檢查是否提供了帳號和密碼
        if username and password:
            print("偵測到帳號密碼，將使用 SQL Server 登入模式。")
            auth_part = f"UID={username};PWD={password};"
        else:
            print("未提供帳號密碼，將嘗試使用 Windows 整合式驗證 (Trusted Connection)。")
            auth_part = "Trusted_Connection=yes;"

        conn_str = (
            f"DRIVER={driver};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"{auth_part}"
            "Encrypt=yes;TrustServerCertificate=yes;"
        )
        
        connection = pyodbc.connect(conn_str)
        print("SQL Server 連線成功。")
        return connection
    except Exception as e:
        print(f"SQL Server 連線失敗: {e}")
        return None

def _connect_mysql():
    """建立並回傳一個 MySQL 的連線物件。"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_MYSQL_PASSWORD"),
            database=os.getenv("DB_MYSQL_NAME")
        )
        print("MySQL 連線成功。")
        return connection
    except Exception as e:
        print(f"MySQL 連線失敗: {e}")
        return None

# 將資料庫類型對應到其專屬的連線函式
DB_CONNECTORS = {
    "sqlserver": _connect_sql_server,
    "mysql": _connect_mysql,
}

def get_db_connection():
    """
    根據 .env 設定，動態選擇並建立資料庫連線。
    """
    db_type = os.getenv("DB_TYPE", "sqlserver").lower()
    connector_func = DB_CONNECTORS.get(db_type)
    
    if not connector_func:
        print(f"錯誤：不支援的資料庫類型 '{db_type}'。請在 .env 中設定 DB_TYPE 為 'sqlserver' 或 'mysql'。")
        return None
        
    print(f"正在嘗試連線到資料庫，類型: {db_type.upper()}")
    return connector_func()

# --- 主要遷移邏輯 ---

def migrate_data():
    """從指定的資料庫讀取資料，產生向量，並儲存到 Qdrant。"""
    # 1. 設定 Gemini
    print("正在設定 Gemini API...")
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Gemini 設定失敗: {e}")
        return

    # 2. 取得資料庫連線
    db_connection = get_db_connection()
    if not db_connection:
        return
        
    # 3. 從 menu_items 資料表取得不重複的菜名
    print("正在從資料庫取得不重複的菜名...")
    cursor = db_connection.cursor()
    cursor.execute("SELECT DISTINCT item_name FROM menu_items;")
    unique_items = [item[0] for item in cursor.fetchall()]
    db_connection.close()
    print(f"找到 {len(unique_items)} 個不重複的菜名。")
    if not unique_items:
        print("資料庫中沒有找到任何菜名，程式結束。")
        return

    # 4. 使用 Gemini 將菜名批次轉換為向量
    print("正在使用 Gemini 將菜名轉換為向量...")
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=unique_items,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = result['embedding']
        print("向量轉換成功。")
    except Exception as e:
        print(f"Gemini 向量轉換失敗: {e}")
        return

    # 5. 連線到 Qdrant 並清空/重建 Collection
    print("正在連線到 Qdrant...")
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # 使用 recreate_collection 來確保每次執行都是從一個乾淨的狀態開始。
        # 這個指令會先刪除同名的舊 Collection (如果存在)，然後再建立一個新的。
        # 這完全符合「先清掉資料，再重新寫入」的需求。
        print(f"正在清空並重建 Qdrant Collection '{COLLECTION_NAME}'...")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' 已成功清空並重建。")
    except Exception as e:
        print(f"Qdrant 操作失敗: {e}")
        return

    # 6. 準備資料點並批次上傳至 Qdrant
    print("正在準備資料點並上傳至 Qdrant...")
    points_to_upsert = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"item_name": item_name}
        ) for i, item_name in enumerate(unique_items)
    ]
    
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert,
            wait=True
        )
        print(f"成功上傳 {len(points_to_upsert)} 個資料點至 Qdrant。")
    except Exception as e:
        print(f"上傳資料至 Qdrant 失敗: {e}")
        return

    print("\n資料遷移完成！")

if __name__ == "__main__":
    migrate_data()
