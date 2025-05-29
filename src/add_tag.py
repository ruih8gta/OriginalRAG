import os
import sys
import configparser

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import json

import model_settings
import common_func
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

asset_path = config['asset']['ASSET_LABEL_PATH']

ASSET_PATH = os.path.join(asset_path)
DOC_ADMIN_PATH = os.path.join(parent_dir,doc_admin_path)

def add_asset_label(asset_path,text):
    asset_path ="prompts/assetlabel_check.txt"
    with open(asset_path, mode='r', encoding='utf-8') as f:
        LABEL_TEMPLATE = f.read()

    with open(ASSET_PATH, mode='r', encoding='utf-8') as f:
        asset_text = f.read()

    prompt_template = ChatPromptTemplate.from_template(LABEL_TEMPLATE)
    chain = prompt_template | chat_instance | StrOutputParser()
    result = chain.invoke({"content": text, "asset": asset_text})
    return result

def update_admin_file(admin_file_path,add_data):
    # ADMIN_PATH内のファイルをすべて読み込み
    with open(admin_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    #json形式のdataにjsonを追加
    add_json = json.loads(add_data)
    #print(data)
    #print(add_json)
    
    data.update(add_json)
    # jsonファイルに書き込み
    with open(admin_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if(__name__ == "__main__"):
    admin_info = common_func.load_admin_doc(DOC_ADMIN_PATH)

    chat_model_type = "4o"
    model_instance = model_settings.models(chat_model_type)  # model_settings.pyファイルのクラスをインスタンス化
    # AzureOpenAIのインスタンスを作成
    chat_instance = model_instance.llm

    for info in admin_info:
        try:
            admin_file_path, file_name, file_path,db_path = info
            docs = common_func.load_doc(file_path)
            result = add_asset_label(ASSET_PATH,str(docs))
            update_admin_file(admin_file_path,result)
            print(f"Processed {file_name} and updated admin file.")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue