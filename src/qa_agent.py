from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
import model_settings
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages.tool import ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import sys
import configparser  # config.iniファイル
import json
from typing import Literal
from loggings import setup_logger, log_message, handle_exception

from create_vectordb import set_rag_data_with_vector
from create_keyworddb import set_rag_data_with_keyword
import model_settings
# パラメータ

# Flask稼働時の設定ファイル読み込み
#config_path ="./config/config.ini"
# プログラム実行時の設定ファイル読み込み
config_path ="../config/config.ini"

config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8') 

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

log_path = os.path.join(parent_dir,config['log']['LOG_PATH'])

doc_db_path = os.path.join(parent_dir,config['doc']['DB_DIR'],"all")

# ログの設定
logger = setup_logger(log_path)
sys.excepthook = handle_exception
log_message(logger, 'ツール起動', to_stdout=True)




#チャットエージェント用のプロンプト
BOT_SYSINT=(
"あなたは、顧客の質問に答えるカスタマーサポートエージェントです。"
"質問に対して、正確で迅速な回答を提供してください。"
"質問が複数の質問で構成されている場合は、tool_div_queryを使用して質問を分割してください。"
"時間に関する質問は、tool_get_timeを使用して時間を確認してください" 
"質問を分割した後は、分割された質問に対して順番に回答してください。"
"質問が分割できない場合は、元の質問に対して回答してください。"
)
#回答作成エージェント用のプロンプト
ANSWER_SYSINT=(
"次の文脈（context）のみに基づいて質問（question）に答えてください。答えはテキスト形式で出力してください:"
"文脈：{context}"
"質問: {question}"
)

QUESTION_SYSINT=(
"# 命令"
"あなたは質問応答システムです。"
"RAGを用いて質問に回答する前に問い合わせ文の内容を分析します"
"この後問い合わせに関するユーザーとシステムのやり取り(conversation)が与えられるので、会話の経緯から、質問が明確になっているかチェックしてください。"
"質問になっていなかったり、問い合わせ内容が不明確であれば明確にしてほしい点を伝えた上で聞き返す文を(response)に出力してください"
"聞き返しが必要な場合は、Step:チャットと出力し、聞き返しが不要な場合は、Step:検索と出力してください"
"聞き返し文は、質問の内容を明確にするためのものであり、質問の内容を変えないようにしてください。"
"聞き返しが不要な場合は聞き返し文には何も出力せず、最終的なユーザーの質問文をまとめて(question)に出力してください"
"結果はjson形式で出力して下さい。jsonという文字や```は含めないでください"
"# 問い合わせ文: {conversation}"
"# 出力形式"
"{{"
"  \"Step\": \"チャット or 検索\","
"  \"response\": \"必要に応じた聞き返しの内容\","
"  \"question\": \"質問文\""
"}}"
)


RETRIVER_SYSINT=(
"# 命令"
"あなたは質問応答システムです。"
"質問の内容に応じて検索に利用するDBを選択します。"
"質問に固有名詞や明確なキーワードが含まれている場合は、tool_retrive_from_keyworddbを使用してキーワードDBを検索してください。"
"質問が曖昧または自然言語的であったり、類似質問や意味的なマッチングが必要な場合は、tool_retrive_from_vectordbを使用してベクトルDBを検索してください"
#"質問が「誰が」「何と関係しているか」など関係性を問うものやナレッジグラフに基づく質推論が必要であれば、tool_retrive_from_graphdbを使用してグラフDBを検索してください。"
)
# 関数
# State
class State(TypedDict):
    """State representing the customer's order conversation."""
    # The step for creating career advice hearing -> check -> plan -> report 
    question:str
    context: str
    answer: str
    #step: str
    messages: Annotated[list, add_messages]
    # Flag indicating that the order is placed and completed.
    step:str # "チャット" "質問"　"検索"　"回答"　"ツール実行" "終了"

# ノード
def Node_chatbot(state: State) -> State:
    """Node for the chatbot to answer questions."""
    #messages_str = "\n".join([str(msg) for msg in state["messages"]])
    #print(f"State!!{state}")

    if state["messages"]:
        #print(state["messages"])
        step = "質問"
        result = state["messages"][-1]
    else:
        result ="マイグレーションに関する質問を入力してください。\n終わりたければqを入力してください。\n新たな質問をする場合はuを入力してください。" 
        step="チャット"   
    
    #log_message(logger, {str(state)}, to_stdout=False)
    #ロガーに、step question answer を出力
    log_message(logger, f"Step: {step} lastMessage: {result}", to_stdout=False)
 
 
    return {"messages": [result],"step": step} 

def Node_question(state: State) -> State:
    """Node for the chatbot to answer questions."""
    # プロンプト入手
    last_msg = state.get("messages", [])[-1]


    prompt = PromptTemplate(
        variables=["conversation"],
        template=QUESTION_SYSINT,
    )
    chat_instance = prompt | chat_instance_question | StrOutputParser()

    log_message(logger, f"Step: {state['step']} message: {state['messages']}", to_stdout=False)
    #answer = chat_instance.invoke({"question": last_msg.content})
    answer = chat_instance.invoke({"conversation": state["messages"]}) 
 
    #結果をjson形式で取得
    q_=""
    answer_json = json.loads(answer)
    #kaprint(answer_json)

    if(answer_json["Step"] == "チャット"):
        step= "チャット"

    elif(answer_json["Step"] == "検索"):
        step = "検索"
        q_= answer_json["question"]
    else:#このパターンは原則なし
        step = "質問"
        
    #log_message(logger, {str(state)}, to_stdout=False)
    log_message(logger, f"Step: {step} lastMessage: {answer_json['response']}", to_stdout=False)

    return {"question": q_,"step": step,"messages": [answer_json["response"]]}

def Node_retriver(state: State) -> State:
    inf_param_instance = model_settings.inf_param()  # model_settings.pyファイルのクラスをインスタンス化
    
    topk = inf_param_instance.Top_k

    question = state.get("question")

    print(f"質問:{question}")
    # 質問をRAGで検索し、結果を取得する
    tool_msg  = chat_instance_retriver_with_tools.invoke(RETRIVER_SYSINT + question)

    if not hasattr(tool_msg, "tool_calls") or len(tool_msg.tool_calls) == 0:
        raise NotImplementedError(f'Do not call any tool in this Question:{question}')
    for tool_call in tool_msg.tool_calls:
        print(f"ツール呼び出し：{tool_call['name']}")
        log_message(logger, f"Step: {state['step']} lastToolMessage: {tool_call}", to_stdout=False)

        if(tool_call["name"] == "tool_retrive_from_vectordb"):
            context = tool_retrive_from_vectordb(question)
        elif(tool_call["name"] == "tool_retrive_from_keyworddb"):
            context = tool_retrive_from_keyworddb(question)
            #elif(tool_call["name"] == "tool_retrive_from_graphdb"):
            #    #仮↓
            #    context = tool_retrive_from_vectordb(question)
        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')
    #print("context:", context)  # 変換されたグラフドキュメントを表示

    for i in range(topk):
        #try:
        #page_num = str(int(context[i].metadata["page"]) + 1)
        log_message(logger,f"【Source_{i+1}】: ")
        #log_message(logger,f" - 文書ファイル名 (ページ番号) : "+context["context"][i].metadata["source"]+" ("+page_num+")")
        log_message(logger,f" - 文書ファイル名 : "+context[i].metadata["source"])
        text = context[i].page_content
        text = text.replace("\n", "")
        text = text.replace("\t", "")
        log_message(logger,f" - テキスト(文頭40字): "+text[:40]+" …") 
    #except:
        #    log_message(logger, "Error: ", to_stdout=True)

    step = "回答"

    #log_message(logger, {str(state)}, to_stdout=False)
    log_message(logger, f"Step: {step}", to_stdout=False)

    return {"context": context,"step": step}

def format_docs(docs:str):
    return "\n\n".join(doc.page_content for doc in docs)

def Node_answer(state: State) -> State:
    """Node for the chatbot to answer questions."""
    # プロンプト入手
    context_ = str(state.get("context"))

    question_ = state.get("question")
    #print(f"質問:{question_}")
    #print(f"文脈:{context_}")

    prompt = PromptTemplate(
        variables=["context", "question"],
        template=ANSWER_SYSINT,
    )
    chat_instance_with_rag = prompt | chat_instance | StrOutputParser()

    answer = chat_instance_with_rag.invoke({"context": context_, "question": question_}) 
    #answer = rag_chain_with_source.invoke({"context": context_, "question": question_})
    # 
    #print(answer)  # 変換されたグラフドキュメントを表示
    step = "チャット"

    #log_message(logger, {str(state)}, to_stdout=False)
    log_message(logger, f"Step: {step} lastMessage: {answer}", to_stdout=False)

    
    return {"answer": answer,"messages": [answer],"step": step}

"""
def Node_tool(state: State) -> State:
    #Node for the chatbot to answer questions.
    tool_msg = state.get("messages", [])[-1]
    outbound_msgs = []
    for tool_call in tool_msg.tool_calls:
        print(f"ツール呼び出し：{tool_call}")

        if(tool_call["name"] == "tool_get_time"):
            result = tool_get_time(tool_call["args"])
        elif(tool_call["name"] == "tool_div_query"):
            result = tool_div_query(tool_call["args"])
        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')
        outbound_msgs.append(
                ToolMessage(
                    content=result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
    return {"messages": outbound_msgs}
"""
def Node_human(state: State) -> State:
    last_msg = state["messages"][-1]
    print("System:", last_msg.content)
    user_input = input("User: ")
    # If it looks like the user is trying to quit, flag the conversation
    # as over.
    if user_input in {"q"}:
        state["step"] = "終了"
    elif user_input in {"u"}:
        user_input=""
        #state["messages"]=[]
        #stateを全て初期化→不具合あり
        return {"messages": []}
    else:
        pass

    #log_message(logger, {str(state)}, to_stdout=False)
    log_message(logger, f"Step: {state['step']} lastMessage: {last_msg}", to_stdout=False)

    return state|{"messages": [("user", user_input)]}

# 条件付きエッジ
def maybe_route_to_tools(state: State) -> Literal["Node_tool", "Node_human"]:
    """Route between chat and tool nodes if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    msg = msgs[-1]

    if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        return "Node_tool"
    else:
        return "Node_human"

def maybe_chat_to_next(state: State) -> Literal["Node_question", "Node_human"]:
    step = state.get("step")
    if(step == "質問"):
        return "Node_question"
    elif(step == "チャット"):
        return "Node_human"
    else:
        raise ValueError(f"Unknown step: {step}")
    
def maybe_question_to_next(state: State) -> Literal["Node_retriver", "Node_human"]:
    step = state.get("step")
    if(step == "検索"):
        return "Node_retriver"
    elif(step == "チャット"):
        return "Node_human"
    else:
        raise ValueError(f"Unknown step: {step}")
    
def maybe_exit_human_node(state: State) -> Literal["Node_chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("step") == "終了":
        return END
    else:
        return "Node_chatbot"

# ツール
@tool
def tool_div_query(query: str) -> str:
    """質問をサブ質問に分割するツール"""
    model_instance_query = model_settings.models(chat_model_type) 
    chat_instance_query = model_instance_query.llm
    template=(
        "以下の質問をサブ質問に分割してください。分割した質問はカンマ区切りで区切ってください。\n"
        "質問: {query}\n"
    )
    prompt = ChatPromptTemplate.from_template(template)

    # LCELによるチェーン作成
    rag_chain_from_data = (
        RunnablePassthrough.assign()
        | prompt
        | chat_instance_query
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"query": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_data)

    # チャット実行
    result = rag_chain_with_source.invoke(query)
    return result


@tool
def tool_retrive_from_vectordb(question) -> list[Document]:
    """質問文をもとにベクトルDBから文書を取得するツール"""
    # ここにベクトルDBから文書を取得するロジックを実装
    inf_param_instance = model_settings.inf_param()
    topk = inf_param_instance.Top_k
    context = set_rag_data_with_vector(question,doc_db_path,emb,topk)
    return context

@tool
def tool_retrive_from_keyworddb(question) -> list[Document]:
    """質問文をもとにキーワードDBから文書を取得するツール"""
    # ここにキーワードDBから文書を取得するロジックを実装
    inf_param_instance = model_settings.inf_param()
    topk = inf_param_instance.Top_k
    context = set_rag_data_with_keyword(question,doc_db_path,topk)
    return context
if(__name__ == "__main__"):
    chat_model_type = "4o"
    emb_model_type = "azure_emb"
    model_instance = model_settings.models(chat_model_type)  # model_settings.pyファイルのクラスをインスタンス化
    # AzureOpenAIのインスタンスを作成
    chat_instance = model_instance.llm
    chat_instance_question =  model_settings.models(chat_model_type).llm

    chat_instance_retriver =  model_settings.models(chat_model_type).llm
    tools = [tool_retrive_from_vectordb,tool_retrive_from_keyworddb]
    chat_instance_retriver_with_tools = chat_instance_retriver.bind_tools(tools)


    emb_instance = model_settings.embeddings(emb_model_type)  # model_settings.pyファイルのクラスをインスタンス化
    emb = emb_instance.emb


    graph_builder = StateGraph(State)
    #ノード追加
    graph_builder.add_node("Node_chatbot", Node_chatbot) # グラフにNodeを追加
    graph_builder.add_node("Node_human", Node_human)
    #graph_builder.add_node("Node_tool", Node_tool)
    graph_builder.add_node("Node_question", Node_question)
    graph_builder.add_node("Node_retriver", Node_retriver)
    graph_builder.add_node("Node_answer", Node_answer)
    #エッジ追加
    graph_builder.add_edge(START, "Node_chatbot")
    #graph_builder.add_edge("Node_question","Node_retriver")
    graph_builder.add_edge("Node_retriver","Node_answer")
    graph_builder.add_edge("Node_answer","Node_human")
    
    #条件付きエッジ
    graph_builder.add_conditional_edges("Node_chatbot", maybe_chat_to_next)
    graph_builder.add_conditional_edges("Node_human", maybe_exit_human_node)
    graph_builder.add_conditional_edges("Node_question", maybe_question_to_next)
    
    app = graph_builder.compile()

    graoh_text = "```mermaid\n"+app.get_graph().draw_mermaid() +"```"## mermaidでCUIに表示する場合
    with open("./tmp/graph.md", "w") as f:
        f.write(graoh_text)
    config = {"recursion_limit": 100}
    state = app.invoke({"messages": []}, config)