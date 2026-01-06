import streamlit as st

st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")

# 使用 markdown 實現置中對齊
st.markdown("<h1 style='text-align: center;'>114(上)『智慧機械設計』課程期末報告</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>指導老師：周榮源</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>班級：四設四甲</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>組別：第二組</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>組員：</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>41123108 仲唯岱</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>41123111 李祀晉</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>41123128 許智堯</h2>", unsafe_allow_html=True)


# 不需要在這裡定義實驗項目列表，因為Streamlit會自動生成側邊欄

# 顯示並排圖片
col1, col2 = st.columns(2)
with col1:
    st.image("files/主題說明.png")
with col2:
    st.image("files/001.jpg")
# Caption
st.caption("這是本報告的主題說明")



