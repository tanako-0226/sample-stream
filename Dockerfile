# Pythonのイメージ
FROM python:3.9

# 作業ディレクトリを設定する
WORKDIR /

# 必要なパッケージをインストールする
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://huggingface.co/datasets/baobab-trees/wikipedia-human-retrieval-ja

EXPOSE 8000
# ローカルのソースコードをコピーする
COPY . .

# コンテナ起動時に実行するコマンドを設定する
WORKDIR /llm_app
# Dockerコンテナのデーモンの開始
# CMD ["streamlit", "run", "app.py"]
CMD python -m streamlit run app.py   --server.port 8000 --server.address 0.0.0.0
