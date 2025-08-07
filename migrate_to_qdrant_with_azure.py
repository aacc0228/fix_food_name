import os
import uuid
import pyodbc
import mysql.connector
from openai import AzureOpenAI
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# --- 設定 ---
load_dotenv()

# Azure OpenAI 設定
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Qdrant 設定
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "taiwan_food_menu_azure"

# 初始化 Azure OpenAI Client
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# --- 通用資料庫連線邏輯 ---
def _connect_sql_server():
    try:
        driver = os.environ.get('DB_DRIVER', '{ODBC Driver 17 for SQL Server}')
        server = os.environ.get('DB_SERVER', 'localhost')
        database = os.environ.get('DB_DATABASE', 'Menu')
        username = os.environ.get('DB_UID')
        password = os.environ.get('DB_PWD')
        auth_part = f"UID={username};PWD={password};" if username and password else "Trusted_Connection=yes;"
        print(f"將使用 {'SQL Server 登入' if username and password else 'Windows 整合式驗證'} 模式。")
        conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};{auth_part}Encrypt=yes;TrustServerCertificate=yes;"
        connection = pyodbc.connect(conn_str)
        print("SQL Server 連線成功。")
        return connection
    except Exception as e:
        print(f"SQL Server 連線失敗: {e}")
        return None

def _connect_mysql():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_MYSQL_PASSWORD"), database=os.getenv("DB_MYSQL_NAME")
        )
        print("MySQL 連線成功。")
        return connection
    except Exception as e:
        print(f"MySQL 連線失敗: {e}")
        return None

DB_CONNECTORS = {"sqlserver": _connect_sql_server, "mysql": _connect_mysql}

def get_db_connection():
    db_type = os.getenv("DB_TYPE", "sqlserver").lower()
    connector_func = DB_CONNECTORS.get(db_type)
    if not connector_func:
        print(f"錯誤：不支援的資料庫類型 '{db_type}'。")
        return None
    print(f"正在嘗試連線到資料庫，類型: {db_type.upper()}")
    return connector_func()

# --- 主要遷移邏輯 ---
def migrate_data():
    """從資料庫讀取資料，用 Azure OpenAI 產生向量，並上傳到 Qdrant。"""
    
    # 1. 從資料庫取得菜名
    db_connection = get_db_connection()
    if not db_connection: return
    
    print("正在從資料庫取得不重複的菜名...")
    cursor = db_connection.cursor()
    cursor.execute("SELECT DISTINCT item_name FROM menu_items;")
    unique_items = [item[0] for item in cursor.fetchall()]
    db_connection.close()
    print(f"找到 {len(unique_items)} 個不重複的菜名。")
    if not unique_items: return

    # 2. 使用 Azure OpenAI 將菜名轉換為向量
    print("正在使用 Azure OpenAI 將菜名轉換為向量...")
    try:
        response = openai_client.embeddings.create(input=unique_items, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
        embeddings = [item.embedding for item in response.data]
        # 取得向量維度，例如 text-embedding-ada-002 是 1536
        vector_size = len(embeddings[0])
        print(f"向量轉換成功，維度: {vector_size}。")
    except Exception as e:
        print(f"Azure OpenAI 向量轉換失敗: {e}")
        return

    # 3. 連線到 Qdrant 並清空/重建 Collection
    print("正在連線到 Qdrant 並準備 Collection...")
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' 已成功清空並重建。")
    except Exception as e:
        print(f"Qdrant 操作失敗: {e}")
        return

    # 4. 準備資料點並上傳至 Qdrant
    print("正在準備資料點並上傳至 Qdrant...")
    points_to_upsert = [
        models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"item_name": name})
        for name, vector in zip(unique_items, embeddings)
    ]
    
    try:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert, wait=True)
        print(f"成功上傳 {len(points_to_upsert)} 個資料點至 Qdrant。")
    except Exception as e:
        print(f"上傳資料至 Qdrant 失敗: {e}")
        return

    print("\n資料遷移至 Qdrant 完成！")

if __name__ == "__main__":
    migrate_data()
