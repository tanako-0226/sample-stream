import streamlit as st
import plotly.graph_objects as go

#resultが追加された場合は、resultsとnamesにも追記

# resultのデータ
result = {'faithfulness': 0.8800, 'answer_relevancy': 0.8634, 'context_precision': 0.9500, 'answer_correctness': 0.5231, 'context_recall': 1.0000}
# result2のデータ
result2 = {'faithfulness': 0.6667, 'answer_relevancy': 0.9437, 'context_precision': 0.8500, 'answer_correctness': 0.4341, 'context_recall': 0.9000}
# result3のデータ
result3 = {'faithfulness': 0.9444, 'answer_relevancy': 0.9486, 'context_precision': 0.9000, 'answer_correctness': 0.4233, 'context_recall': 0.8000}


# グラフ作成
fig = go.Figure()

# データをリスト化
results = [result, result2, result3]
names = ['Result 1', 'Result 2', 'Result 3']
colors = ['blue', 'orange', 'green']  # データごとの色

# データを追加
for i, res in enumerate(results):
    data = {
        'context_precision': res['context_precision'],
        'faithfulness': res['faithfulness'],
        'answer_relevancy': res['answer_relevancy'],
        'context_recall': res['context_recall'],
        'answer_correctness': res['answer_correctness'],
    }
    fig.add_trace(go.Scatterpolar(
        r=list(data.values()),
        theta=list(data.keys()),
        fill='toself',
        name=names[i],  # レジェンドの名前
        #line=dict(color=colors[i])  # グラフの色を指定
    ))

# レイアウト設定
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='Retrieval Augmented Generation - Evaluation',
    width=800,
)

# グラフを表示
st.plotly_chart(fig)
