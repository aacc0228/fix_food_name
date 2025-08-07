import os
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import CountResult
from flask import Flask, render_template, request
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

# --- 初始化 ---
app = Flask(__name__)
openai_client = None
qdrant_client = None

# 初始化 Azure OpenAI Client
try:
    if all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBEDDING_DEPLOYMENT, AZURE_OPENAI_API_VERSION]):
        openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        print("✅ 成功初始化 Azure OpenAI Client。")
    else:
        print("⚠️ 警告：Azure OpenAI 的環境變數不完整，將無法產生向量。")
except Exception as e:
    print(f"❌ 初始化 Azure OpenAI Client 失敗: {e}")

# 初始化 Qdrant Client
try:
    if all([QDRANT_URL, QDRANT_API_KEY]):
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("✅ 成功連線到 Qdrant。")
    else:
        print("⚠️ 警告：Qdrant 的環境變數不完整，將無法進行搜尋。")
except Exception as e:
    print(f"❌ Qdrant 連線失敗: {e}")


# --- 核心搜尋函式 ---
def search_similar_item(query):
    log_parts = [f"使用者查詢: \"{query}\""]
    
    if not openai_client:
        return "錯誤：Azure OpenAI Client 未初始化。", "Client 初始化失敗"
    if not qdrant_client:
        return "錯誤：Qdrant Client 未初始化。", "Client 初始化失敗"

    try:
        # 1. 檢查 Qdrant Collection 狀態
        try:
            collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            point_count = collection_info.points_count
            log_parts.append(f"✅ Qdrant 狀態檢查成功：Collection '{COLLECTION_NAME}' 中共有 {point_count} 筆資料。")
            if point_count == 0:
                return "錯誤：向量資料庫是空的，請先執行資料遷移腳本。", "\n".join(log_parts)
        except Exception as e:
            log_parts.append(f"❌ Qdrant 狀態檢查失敗：無法取得 Collection '{COLLECTION_NAME}' 的資訊。")
            log_parts.append(f"錯誤詳情: {e}")
            return "錯誤：無法連接到向量資料庫的 Collection。", "\n".join(log_parts)

        # 2. 將使用者查詢轉換為向量
        response = openai_client.embeddings.create(input=query, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
        query_vector = response.data[0].embedding
        log_parts.append(f"✅ 查詢向量已產生。")

        # 3. 進行一次無門檻的搜尋
        raw_search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=1 
        )
        
        # 在網頁日誌中加入原始結果
        log_parts.append("\n--- 原始搜尋結果 (無分數門檻) ---")
        log_parts.append(str(raw_search_result))
        log_parts.append("---------------------------------")

        # 4. 根據原始結果進行篩選並整理
        score_threshold = 0.65
        found_items = []
        
        # ★★★ 強化日誌：在伺服器後台印出詳細的篩選過程 ★★★
        print("\n--- [後台日誌] 篩選迴圈檢查 ---")
        for hit in raw_search_result:
            item_score = hit.score
            item_name = hit.payload.get("item_name")
            is_above_threshold = item_score >= score_threshold
            # 在終端機印出每一步的判斷
            print(f"檢查項目: '{item_name}', 分數: {item_score:.4f}, 是否高於門檻({score_threshold})? -> {is_above_threshold}")
            if is_above_threshold:
                found_items.append({"name": item_name, "score": item_score})
                print(f"  -> ✅ 已加入結果列表。目前列表長度: {len(found_items)}")
        print("--- [篩選結束] ---\n")
        
        if not found_items and raw_search_result:
             log_parts.append(f"\n⚠️ 注意：已找到相似項目，但最高分 ({raw_search_result[0].score:.4f}) 未達到門檻 (>= {score_threshold})。")
        
        log_parts.append(f"\n套用門檻 (>= {score_threshold}) 後，最終找到 {len(found_items)} 筆結果。")
        
        return found_items, "\n".join(log_parts)

    except Exception as e:
        error_message = f"搜尋時發生嚴重錯誤: {type(e).__name__}"
        error_log = f"錯誤類型: {type(e).__name__}\n錯誤詳情: {str(e)}"
        print(f"❌ 搜尋錯誤: {error_log}")
        return error_message, error_log

# --- Flask 路由 ---
@app.route('/', methods=['GET', 'POST'])
def index():
    results, log_info, query = None, None, ""
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            results, log_info = search_similar_item(query)
            
    return render_template('index.html', query=query, results=results, log=log_info)

# --- 啟動伺服器 ---
if __name__ == '__main__':
    app.run(debug=True)
