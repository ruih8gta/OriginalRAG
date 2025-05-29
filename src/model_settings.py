import os
import sys
import configparser # config.iniファイル
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
#from langchain_aws import ChatBedrock
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain_aws import BedrockEmbeddings
#import boto3

config = configparser.ConfigParser()
#sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
#sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'backend'))
#config.read('./config/config.ini', encoding='utf-8') #Flask稼働時
config.read('../config/config.ini', encoding='utf-8')


# 推論パラメータ設定
class inf_param:
    # for chat model
    Temperature = 0.0
    Top_p = 1
    Max_tokens = None # 2048など
    Top_k = 10
    # for embedding model
    CHUNK_SIZE = 4096
    CHUNK_OVERLAP = 512
    # for search
    Top_k = 3  # 参照するドキュメント(chunk)数
    Search_distance = 0.0 # 検索距離の閾値

# チャットを利用する場合
class models:
    def __init__(self, chat_model_type):
        if chat_model_type == "4o":
            llm, chat_model_name = models.ChatOpenAI("4o")
        elif chat_model_type == "gpt-4o-mini":
            llm, chat_model_name = models.Azure_Chat_model("Azure-Chat-4o-mini")
        else:
            print("Chatモデルの選択エラー [-cm gpt-4o / gpt-35t] \n")
            sys.exit()
        self.llm = llm
        self.chat_model_name = chat_model_name

    def ChatOpenAI(conf_azure):
        model_name = config[conf_azure]['MODEL']

        ref_temperature = inf_param.Temperature
        chat_model = ChatOpenAI(
            api_key=config[conf_azure]['API_KEY'],
            model_name =config[conf_azure]['MODEL'],
            temperature=ref_temperature,
            top_p=inf_param.Top_p,
            max_tokens=inf_param.Max_tokens,

        )
        return chat_model, model_name

# ベクトル検索を利用する場合
class embeddings:
    def __init__(self, emb_model_type):
        if emb_model_type == "azure_emb":
            emb, emb_model_name = embeddings.Azure_Embedding_model("Azure-Embedding-ada")
        elif emb_model_type == "azure_emb_large":
            emb, emb_model_name = embeddings.Azure_Embedding_model("Azure-Embedding-3-large")
        else:
            print("Embeddingモデルの選択エラー [-em azure_emb] \n")
            sys.exit()
        self.emb = emb
        self.emb_model_name = emb_model_name

    def Azure_Embedding_model(conf_azure):
        model_name= config[conf_azure]['MODEL']
        embeddings = OpenAIEmbeddings(
            api_key=config[conf_azure]['API_KEY'],
            model=config[conf_azure]['MODEL'],
            chunk_size=1,
            disallowed_special=()
        )
        return embeddings, model_name