from langchain.schema import Document
import json
import os

# jsonファイルを読み込み、langchainのドキュメント形式に変換
def load_doc(file_path):
    # JSONファイルを読み込み、JSON形式で登録
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    doc = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in data
    ]
    return doc

# ドキュメント管理情報が書かれたjsonファイルを読み込み、管理情報を返す
def load_admin_doc(ADMIN_PATH):
    #ADMIN_PATH内のファイルをすべて読み込み
    docs_info = []
    for cur_dir, inc_dir, file_names in os.walk((ADMIN_PATH)):
        for file_name in file_names:
            file_path = os.path.join(cur_dir, file_name)
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data_admin = json.load(f)
                docs_info.append([file_path,data_admin["file_name"],data_admin["doc_text_file_path"],data_admin["vector_db_path"]])               
            else:
                pass
    return docs_info


# ドキュメント管理情報が書かれたjsonファイルを読み込み、管理情報を返す
def load_admin_doc_key(ADMIN_PATH):
    #ADMIN_PATH内のファイルをすべて読み込み
    docs_info = []
    for cur_dir, inc_dir, file_names in os.walk((ADMIN_PATH)):
        for file_name in file_names:
            file_path = os.path.join(cur_dir, file_name)
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data_admin = json.load(f)
                docs_info.append([file_path,data_admin["file_name"],data_admin["doc_text_file_path"],data_admin["vector_db_path"],data_admin["id"]])               
            else:
                pass
    return docs_info