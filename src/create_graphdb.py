from neo4j import GraphDatabase
import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
#from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
import sys
import json
import configparser  # config.iniファイル

from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document

#sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
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


# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://localhost"
AUTH = ("neo4j", "password")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

def create_graph_doc_file(chat_instance,documents,db_path):
    # テキストファイルを読み込む

     # プロンプトテンプレートの設定
    # graphdb_create.txtを読み込み
    template =""
    with open("./prompts/graphdb_create.txt", "r", encoding="utf-8") as f:
        template = f.read()

    prompt = ChatPromptTemplate.from_template(template)

    # LCELによるチェーン作成
    rag_chain_from_data = (
        RunnablePassthrough.assign()
        | prompt
        | chat_instance
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"document": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_data)

    # チャット実行
    graph_documents = rag_chain_with_source.invoke(documents)
         
    #print(graph_documents['answer'])  # 変換されたグラフドキュメントを表示

    # graph_documentsを

    graph_dict_list = json.loads(f"{graph_documents['answer']}")
    
    # 結果をローカルファイルに保存
    db_file_path = os.path.join(db_path, "graphdb.json")
    print("Save " + db_path + " ...", end="")
    with open(db_file_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict_list, f, ensure_ascii=False, indent=4)
    print("Done")
    return graph_dict_list


def add_node(tx, pair_node_info):
    for name, text  in pair_node_info:
        query = f'CREATE (f:node [name: $name,text: $text] ) RETURN f'
        query = query.replace("[", "{").replace("]", "}")
        tx.run(query, name=name, text=text)

def add_relation(tx, rows):
    for row in rows:
        from_node = row["source"]
        to_node = row["target"]
        relation = row["relation"]
        relation = row["relation"].replace(":", "_").replace(" ", "_")  # コロンとスペースをアンダースコアに置換


        tx.run('MATCH (f1:node {name: $from_node})'
                'MATCH (f2:node {name: $to_node})'
                f'CREATE (f1)-[:{relation}]->(f2)',
                from_node=from_node, relation=relation, to_node=to_node)

def isert_db(data):
    # ドライバの作成とセッションの開始

    # GraphDocumentを作成
    node_list = [i["id"] for i in data["nodes"] ]
    label_list = [i["text"] for i in data["nodes"]]
    pair_node_info = []
    for idx in range(len(node_list)):
        pair_node_info.append((node_list[idx], label_list[idx]))
    pair_node_info = list(set(pair_node_info))

    with driver.session() as session:
        session.execute_write(add_node, pair_node_info)

    relations= [i for i in data["edges"]] 
    with driver.session() as session:
        session.execute_write(add_relation, relations)


def fetch_graph(tx):
    query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25"
    result = tx.run(query)
    return result

def delete_graph(tx):
    query = "MATCH (n) DETACH DELETE n"
    result = tx.run(query)
    return result



def retrive():
    emb_model_type = "azure_emb"
    emb_instance = model_settings.embeddings(emb_model_type)
    index = Neo4jVector.from_existing_graph(
        embedding=emb_instance.emb,
        url = URI,
        username = AUTH[0],
        password = AUTH[1],
        node_label="node", # 検索対象ノード
        text_node_properties=["id","name","text"], # 検索対象プロパティ 
        embedding_node_property="embedding", # ベクトルデータの保存先プロパティ
        index_name="vector_index", # ベクトル検索用のインデックス名
        keyword_index_name="person_index", # 全文検索用のインデックス名
        search_type="hybrid" # 検索タイプに「ハイブリッド」を設定（デフォルトは「ベクター」）
    )
    query = "裏切り"
    docs_with_score = index.similarity_search_with_score(query, k=3)
    # 検索結果の表示
    for doc, score in docs_with_score:
        print(doc.page_content)
        print(f"スコア: {score}\n")

def rag(query_prompt):
    model_instance = model_settings.models("4o")  # model_settings.pyファイルのクラスをインスタンス化
    llm = model_instance.llm 
    # Cypherクエリ用のプロンプトテンプレート
    ## 素のlangchainテンプレートだと表記ゆれに弱いので、自前で少し修正(id_list部分の記述を追加)
    CYPHER_GENERATION_TEMPLATE = """
    Task: グラフデータベースに問い合わせるCypher文を生成する。

    指示:
    schemaで提供されている関係タイプとプロパティのみを使用してください。
    提供されていない他の関係タイプやプロパティは使用しないでください。
    ノードで指定するidはid_listで提供されているidの中から近いものを選択し、変更してください。

    schema:
    {schema}
    
    id_list:
    {id_list}

    注意: 回答に説明や謝罪は含めないでください。
    Cypher ステートメントを作成すること以外を問うような質問には回答しないでください。
    生成された Cypher ステートメント以外のテキストを含めないでください。

    例) 以下は、特定の質問に対して生成されたCypher文の例です:
    # モアナ2の主役俳優の1人は?
    MATCH (e1:__Entity__)-[:STARS]->(e2:__Entity__) WHERE e1.id = 'アナと伝説の海2' RETURN e2.id
    # 頭文字Dの作者は?
    MATCH (e1:__Entity__)-[:AUTHOR]->(e2:__Entity__) WHERE e1.id = '頭文字D' RETURN e2.id
    # モアナ2の制作会社は?
    MATCH (e1:__Entity__)-[:PRODUCED]->(e2:__Entity__) WHERE e1.id = 'Moana 2' RETURN e2.id

    質問: {question}"""

    graph = Neo4jGraph(
        url = URI,
        username = AUTH[0],
        password = AUTH[1],
        )
    # プロンプトテンプレートからプロンプトを作成
    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], 
        template=CYPHER_GENERATION_TEMPLATE
    ).partial(id_list=id_query())

    # Cypherクエリを作成 → 実行 → 結果から回答を行うChainを作成
    cypher_chain = GraphCypherQAChain.from_llm(
        llm = llm,
        graph=graph,
        cypher_prompt=CYPHER_GENERATION_PROMPT, # Cypherクエリ用にプロンプトをセット
        verbose=True, # 詳細表示を「True」に設定
        allow_dangerous_requests=True,
        return_intermediate_steps=True, # 中間ステップを返す場合
    )

    # 質問文を設定してllmから回答を取得
    result = cypher_chain.invoke({'query': query_prompt})
    
    # 回答を表示
    #print(result)
    print(f"質問: {query_prompt}")
    print(f"Intermediate steps: {result['intermediate_steps']}") # return_intermediate_steps=True
    print(f"Final answer: {result['result']}")
    #print(graph.schema)

# グラフDBのノードのidプロパティ確認
def id_query():
    graph = Neo4jGraph(
        url = URI,
        username = AUTH[0],
        password = AUTH[1],
    )
    # DB内のグラフを削除するクエリ（複数回実行用）
    id_cypher = """
    MATCH (n:__Entity__)
    RETURN n.id
    """
    result_id = graph.query(id_cypher)
    return result_id    


# グラフDBのデータをJSON形式でエクスポートする関数
def export_data(tx):
    query = """
    CALL apoc.export.json.all('exported_data.json', {useTypes:true})
    """
    tx.run(query)

if __name__ == "__main__":

    DOC_ADMIN_PATH = os.path.join(parent_dir,doc_admin_path)

    chat_model_type = "4o"
    model_instance = model_settings.models(chat_model_type)  # model_settings.pyファイルのクラスをインスタンス化
    # AzureOpenAIのインスタンスを作成
    chat_instance = model_instance.llm

    admin_info = common_func.load_admin_doc(DOC_ADMIN_PATH)

    for ad,file_name, file_path,db_path in admin_info:
        try:   
            docs = common_func.load_doc(file_path)
            data = create_graph_doc_file(chat_instance,docs,db_path)
            isert_db(data)
        except Exception as e:
            print(f"Error processing file: {file_name} - {str(e)}")
    print("Completed")
    #
    #show()
    #retrive()
    #query_prompt ="勘定系システムを構成する要素について教えて"
    #rag(query_prompt)
    #export_graph_data()
    #import_graph_data("exported_data.json")