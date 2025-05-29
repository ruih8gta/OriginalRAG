# Azure埋め込みモデルでデータをベクトル化
import os
import sys
import argparse
import configparser  # config.iniファイル
import openai  # OpenAI APIのインポート
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import json

#sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'backend')) # backendフォルダをパスに追加
#sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import model_settings
import common_func
# 推論パラメータ設定


# Flask稼働時の設定ファイル読み込み
#config_path ="./config/config.ini"
# プログラム実行時の設定ファイル読み込み
config_path ="../config/config.ini"

config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8') 


doc_admin_path = config['doc']['DOC_ADMIN_DIR']
doc_db_path = config['doc']['DB_DIR']

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)


# ベクトルDB作成
def Create_vectorDB_index(embeddings, documents, VDB_PATH):
    inf_param_instance = model_settings.inf_param()
    if not documents:  # documentsが空の場合のチェック
        raise ValueError("No documents to process. Please check the input data.")
    text_splitter = CharacterTextSplitter(
        chunk_size=inf_param_instance.CHUNK_SIZE,
        chunk_overlap=inf_param_instance.CHUNK_OVERLAP,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:  # chunksが空の場合のチェック
        raise ValueError("No chunks created from documents. Please check the input data.")
    print("Creating vectorDB...", end="")
    db = FAISS.from_documents(chunks, embeddings)
    print("Done")
    print("Save " + VDB_PATH + " ...", end="")
    db.save_local(VDB_PATH)
    print("Done")


# 質問内容から関係するドキュメントを検索する関数
def set_rag_data_with_vector(question,db_path,emb,topk):
    """Set the RAG data for the question."""
        
    # ローカルに保存したFAISS-ベクトルDB読み込み
    db = FAISS.load_local(
        db_path, 
        emb,
        allow_dangerous_deserialization=True
    )
    # 検索機能の設定
    retriever=db.as_retriever(
        search_kwargs={
            'k': topk, # 検索した情報の上位何件までLLMに渡すか
        }
    )
    # 質問をRAGで検索し、結果を取得する
    context = retriever.get_relevant_documents(question)
    
    return context



# 選択した埋め込みモデルでデータをベクトル化する
if __name__ == "__main__":

    DOC_ADMIN_PATH = os.path.join(parent_dir,doc_admin_path)

    emb_model_type = "azure_emb"
    emb_instance = model_settings.embeddings(emb_model_type)
    emb = emb_instance.emb
    
    admin_info = common_func.load_admin_doc(DOC_ADMIN_PATH)
    all_doc=[]

    for ad,file_name, file_path,db_path in admin_info:    
        docs = common_func.load_doc(file_path)
        all_doc.extend(docs)
        Create_vectorDB_index(emb, docs, db_path)
    print("Completed")
    all_db_path =os.path.join(parent_dir,doc_db_path,"all")
    Create_vectorDB_index(emb, all_doc, all_db_path)
    print("Completed")

