import math
from collections import Counter

from janome.tokenizer import Tokenizer
from langchain_core.documents import Document
import configparser  # config.iniファイル
import os
import common_func
import json

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



# Sample documents
samnple_documents = [
    "Python is a popular programming language.",
    "RAG combines retrieval and generation for better answers.",
    "BM25 is a ranking function used by search engines.",
    "OpenAI GPT models are powerful for text generation.",
    "LangChain and LangGraph are useful for building workflows."
]


class BM25:
    """
    BM25（Best Matching 25）アルゴリズムの実装クラス
    文書コレクション内でのキーワード検索と関連性スコアリングを行う
    """
    def __init__(self, documents, doc_ids=None):
        """
        BM25クラスの初期化
        
        Args:
            documents: 文書のリスト（各文書は文字列）
        """
        self.documents = documents  # 文書コレクション
        self.corpus_size = len(documents)  # コーパス（文書集合）のサイズ
        self.avgdl = sum(len(doc.split()) for doc in documents) / self.corpus_size  # 平均文書長
        self.k1 = 1.5  # 単語頻度の正規化パラメータ
        self.b = 0.75  # 文書長正規化パラメータ
        self.f = []  # 各文書内の単語頻度を格納するリスト
        self.df = {}  # 単語の文書頻度（その単語が出現する文書数）
        self.idf = {}  # 逆文書頻度（単語の重要度を表す指標）
        # 文書IDが指定されていればそれを使用、なければ連番を生成
        self.doc_ids = doc_ids if doc_ids is not None else list(range(len(documents)))
        # IDと文書の数が一致していることを確認
        if len(self.doc_ids) != len(documents):
            raise ValueError("文書IDの数が文書の数と一致しません。")
        
        self.initialize()  # 頻度とIDFの初期化

    def initialize(self):
        """
        BM25に必要な統計情報（単語頻度、文書頻度、逆文書頻度）を計算
        """
        # 各文書の単語頻度と文書頻度を計算
        for document in self.documents:
            frequencies = Counter(document.split())  # 文書内の単語頻度をカウント
            self.f.append(frequencies)  # 単語頻度を保存
            
            # 単語の文書頻度を更新
            for word, freq in frequencies.items():
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1  # その単語が出現する文書数をカウント
        
        # 逆文書頻度(IDF)を計算
        for word, freq in self.df.items():
            # BM25の逆文書頻度計算式
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def score(self, query, index):
        """
        クエリと特定の文書のBM25スコアを計算
        
        Args:
            query: 検索クエリ（文字列）
            index: スコアを計算する文書のインデックス
            
        Returns:
            float: BM25スコア（高いほど関連性が高い）
        """
        score = 0.0
        doc = self.documents[index]
        frequencies = self.f[index]  # 文書内の単語頻度
        doc_len = len(doc.split())  # 文書の長さ（単語数）
        
        # クエリ内の各単語について、BM25スコアを計算
        for word in query.split():
            if word in frequencies:
                freq = frequencies[word]  # 文書内のその単語の頻度
                # BM25スコア計算式
                score += (self.idf[word] * freq * (self.k1 + 1) /
                         (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_scores(self, query):
        """
        クエリに対するすべての文書のスコアを計算し、降順でソートして返す
        
        Args:
            query: 検索クエリ（文字列）
            
        Returns:
            list: (スコア, 文書ID, 文書)のタプルのリスト（スコア降順）
        """
        scores = []
        # 全ての文書について、クエリに対するスコアを計算
        for index in range(self.corpus_size):
            score = self.score(query, index)
            scores.append((score, self.doc_ids[index], self.documents[index]))
        
        # スコアの高い順にソート
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores


def tokenizer_func(text):
   tokenizer = Tokenizer()
   # 分かち書き（単語ごとにスペースで区切る）
   #wakachi = [token.surface for token in tokenizer.tokenize(text)]
   wakachi = [token.surface for token in tokenizer.tokenize(text) if token.part_of_speech.startswith("名詞")]

   result = " ".join(wakachi)
   return result

def save_keyword_db(doc_list, db_path):
    # データベースに保存する処理を実装
    # ここでは例として、データをテキスト形式で保存する
    print("Save " + db_path + " ...", end="")
    with open(os.path.join(db_path,"keyword.txt"), "w", encoding="utf-8") as f:
        for item in doc_list:
            f.write(item + "\n")
    print("Done")

def save_keyword_db_json(doc_list, db_path):
    # データベースに保存する処理を実装
    # ここでは例として、データをJSON形式で保存する
    print("Save " + db_path + " ...", end="")
    with open(os.path.join(db_path,"keyword.json"), "w", encoding="utf-8") as f:
        json.dump(doc_list, f, ensure_ascii=False, indent=4)
    print("Done")

def search_keyword_db(query, db_path):
    # Initialize BM25 with documents
    documents, doc_ids = load_keyword_db(db_path)
    bm25 = BM25(documents, doc_ids=doc_ids)

    query_token = tokenizer_func(query)

    # Get scores for the query
    scores = bm25.get_scores(query_token)

    # Display top 10 results
    print(f"Query: {query}")
    top_10_scores = scores[:10]
    for score, doc_id, doc in top_10_scores:
        # 文書IDを使用してファイル名を取得
        idx = doc_ids.index(doc_id) if doc_id in doc_ids else -1
        print(f"Score: {score:.4f}, Document ID: {doc_id}, Preview: {doc[:20]}")

def load_keyword_db(db_path):
    # データベースからドキュメントを読み込む処理を実装
    if db_path.endswith(".json"):
        with open(db_path, "r", encoding="utf-8") as f:
            keyword_data = json.load(f)

    documents = [item["keywords"] for item in keyword_data]
    doc_ids = [item["id"] for item in keyword_data]
    return documents, doc_ids



def create_keyword_db(admin_path):
    admin_info = common_func.load_admin_doc_key(admin_path)

    documents =[]
    doc_ids = []
    doc_names = []

    for ad,file_name, file_path,db_path,id in admin_info:    
        #　ドキュメント形式で読み込み
        wakachi_docs = ""
        docs = common_func.load_doc(file_path)
        for doc in docs:
            #　分かち書き
            wakachi = tokenizer_func(doc.page_content)
            wakachi_docs += wakachi + " "
        save_keyword_db([wakachi_docs], db_path)

        documents.append(wakachi_docs)
        doc_ids.append(id)
        doc_names.append(file_name)

    doc_db_path="C:\\Users\\nakagawa\\Desktop\\Knowledge_RAG\\data\\db"
    all_db_path =os.path.join(doc_db_path,"all")  
    # 全てのドキュメントを一つのDBに保存
    #フォルダがなければ作成
    if not os.path.exists(all_db_path):
        os.makedirs(all_db_path, exist_ok=True)
    data_json =[]
    for id, doc in zip(doc_ids,documents):
        data_json.append({"id": id, "keywords": doc}) 
    save_keyword_db_json(data_json, all_db_path)


def set_rag_data_with_keyword(question,db_path,topk):
    # Initialize BM25 with documents
    db_path=   os.path.join(db_path,"keyword.json")
    documents, doc_ids = load_keyword_db(db_path)
    bm25 = BM25(documents, doc_ids=doc_ids)

    query_token = tokenizer_func(question)

    # Get scores for the query
    scores = bm25.get_scores(query_token)

    context = []

    # Display top 10 results
    #print(f"Query: {question}")
    top_k_scores = scores[:topk]
    for score, doc_id, doc in top_k_scores:
        # 文書IDを使用してファイル名を取得
        idx = doc_ids.index(doc_id) if doc_id in doc_ids else -1
        #print(f"Score: {score:.4f}, Document ID: {doc_id}, Preview: {doc[:20]}")
        context.append(Document(id=doc_id, metadata={"source": str(doc_id)}, page_content=doc))
    return context


if __name__ == "__main__":

    DOC_ADMIN_PATH = os.path.join(parent_dir,doc_admin_path)
    create_keyword_db(DOC_ADMIN_PATH)
    query = "過去の不具合事例について教えて"
    doc_db_path_key="C:\\Users\\nakagawa\\Desktop\\Knowledge_RAG\\data\\db\\all\\keyword.json"
    search_keyword_db(query, doc_db_path_key)
