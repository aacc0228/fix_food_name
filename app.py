import os
import json # 匯入 json 模組
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from flask import Flask, render_template, request, jsonify, Response # 匯入 Response
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
# 雖然我們將手動處理，但保留此設定是個好習慣
app.config['JSON_AS_ASCII'] = False

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
        print("⚠️ 警告：Azure OpenAI 的環境變數不完整。")
except Exception as e:
    print(f"❌ 初始化 Azure OpenAI Client 失敗: {e}")

# 初始化 Qdrant Client
try:
    if all([QDRANT_URL, QDRANT_API_KEY]):
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("✅ 成功連線到 Qdrant。")
    else:
        print("⚠️ 警告：Qdrant 的環境變數不完整。")
except Exception as e:
    print(f"❌ Qdrant 連線失敗: {e}")

# --- 核心搜尋函式 (保持不變) ---
def search_similar_item(query):
    if not openai_client or not qdrant_client:
        return "錯誤：後端服務未完全初始化。", "Client 初始化失敗"

    try:
        # 1. 產生查詢向量
        response = openai_client.embeddings.create(input=query, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
        query_vector = response.data[0].embedding

        # 2. 在 Qdrant 搜尋
        # ★★★ 修正點：將 limit 改為 1，只找出最相關的一筆結果 ★★★
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=1,
            score_threshold=0.65
        )

        # 3. 整理結果
        found_items = [{"name": hit.payload.get("item_name"), "score": hit.score} for hit in search_result]
        log = f"Qdrant 原始回傳: {search_result}"
        return found_items, log

    except Exception as e:
        error_message = f"搜尋時發生嚴重錯誤: {type(e).__name__}"
        error_log = f"錯誤詳情: {str(e)}"
        print(f"❌ 搜尋錯誤: {error_log}")
        return error_message, error_log

# --- Flask 路由 ---

# 原有的網頁介面路由 (保持不變)
@app.route('/', methods=['GET', 'POST'])
def index():
    results, log_info, query = None, None, ""
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            results, log_info = search_similar_item(query)
            
    return render_template('index.html', query=query, results=results, log=log_info)

# ★★★ API 路由修改點 ★★★
@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json()
    if not data or 'query' not in data:
        error_response = json.dumps({'error': '請求內容缺少 "query" 欄位'}, ensure_ascii=False)
        return Response(error_response, mimetype='application/json; charset=utf-8', status=400)

    query = data['query']
    results, log = search_similar_item(query)

    if isinstance(results, str) and "錯誤" in results:
        error_response = json.dumps({'error': results, 'log': log}, ensure_ascii=False)
        return Response(error_response, mimetype='application/json; charset=utf-8', status=500)

    # ★★★ 修正點：處理單一回傳結果 ★★★
    # 從結果列表中取出第一個項目，如果列表是空的，則回傳 null
    single_result = results[0] if results else None
    
    # 手動建立一個包含正確編碼標頭的 Response 物件
    response_data = {
        'query': query,
        'result': single_result # 將 'results' 列表改為 'result' 物件
    }
    # 使用 json.dumps 將 python dict 轉換為 JSON 字串，並確保中文不被跳脫
    json_response = json.dumps(response_data, ensure_ascii=False)
    
    # 建立一個 Response 物件，並明確指定 mimetype 和 charset
    return Response(json_response, mimetype='application/json; charset=utf-8')

# --- 啟動伺服器 ---
if __name__ == '__main__':
    app.run(debug=True)
