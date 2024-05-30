import streamlit as st
from openai import AzureOpenAI
import faiss
import numpy as np
import pandas as pd
import random
import ast
import os
from ragas import evaluate 

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

from ragas.metrics import faithfulness, answer_correctness
from datasets import load_dataset, Dataset


metrics = [
    answer_correctness
]

#環境変数設定
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
OPENAI_API_VERSION = "2024-02-01"
AZURE_ENDPOINT =os.getenv('AZURE_OPENAI_ENDPOINT')
MODEL = os.getenv('model')
TEXT_EMB = "text-embedding-ada-002"


azure_model = AzureChatOpenAI(
    openai_api_version = OPENAI_API_VERSION,
    azure_endpoint = AZURE_ENDPOINT,
    azure_deployment = MODEL,
    model = MODEL,
    validate_base_url=False
)

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version = OPENAI_API_VERSION,
    azure_endpoint = AZURE_ENDPOINT,
    azure_deployment = TEXT_EMB,
    model = TEXT_EMB
)


client = AzureOpenAI(
    api_key = AZURE_API_KEY, 
    api_version = OPENAI_API_VERSION,
    azure_endpoint = AZURE_ENDPOINT
)


def make_index(data):
    """
    Create FAISS index.
    """
    index = faiss.IndexFlatIP(1536)
    # Convert the answer embeddings from string representation to list of floats
    tmp = [ast.literal_eval(i) if isinstance(i, str) else i for i in data["answer_emb"].values]
    index.add(np.array(tmp).astype('float32'))
    return index

def generate_llm_answer(df, question_num = 2, seed=42):
    # contextsとanswersの初期化
    answers = []
    ground_truth = []
    questions_list = []

    # 設定可能な質問数
    N = question_num

    # 0から838のリストを作成し、そこからN個をランダムに抽出
    # random.seed(42) #シードを固定したい時
    random_indices = random.sample(range(838), N)

    # ランダムに抽出した質問に対して処理を行う
    for question_num in random_indices:
        messages = [{"role": "system", "content": df["question"][question_num]}]
        
        response = client.chat.completions.create(model = MODEL, 
                                                temperature=0.3,
                                                max_tokens=3000,
                                                messages=messages)

        answers.append(response.choices[0].message.content)
        ground_truth.append(ast.literal_eval(df["answer"][question_num])["text"])
        questions_list.append(df["question"][question_num])

        return answers, ground_truth, questions_list

def calc_rags(answers, ground_truth, questions_list):
    # データセットの準備
    ds = Dataset.from_dict(
        {
            "question": questions_list,
            "answer": answers,
            "ground_truth": ground_truth,
        }
    )

    result = evaluate(
        ds, metrics=metrics, llm=azure_model, embeddings=azure_embeddings, raise_exceptions=True
    )

    print(result)

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("emb_data.csv")

    index = make_index(df)

    df["question_emb"] = df["question_emb"].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
    question_emb_np = np.stack(df["question_emb"].values).astype('float32')

    answers, ground_truth, questions_list = generate_llm_answer(df, question_num=2)
    calc_rags(questions_list, ground_truth, answers)