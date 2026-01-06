import streamlit as st
import pandas as pd
import numpy as np
# import openpyxl
import xlrd

import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode


st.title("第一章 TurtleBot Burger平台")

st.write("TurtleBot Burger平台是一台基於TurtleBot 3平台的機器人平台，它具有以下特點：")



st.header("實驗步驟：")
st.subheader("""實驗步驟""")
# sub-header
st.subheader("""步驟一：VirtualBox 設定""")


col1, col2, col3 = st.columns(3)
with col1:
    st.image("files/008.jpg")
    st.caption("VirtualBox 系統配置")
with col2:
    st.image("files/009.jpg")
    st.caption("VirtualBox 網路設定")
with col3:
    st.image("files/010.jpg")
    st.caption("VirtualBox 環境準備")

st.subheader("""步驟二：安裝 Turtlebot3""")
st.image("files/0030.png")
st.caption("安裝 Turtlebot3")
st.subheader("""步驟三：啟動 /roscore""")
st.image("files/0029.png")
st.caption("啟動 /roscore")

st.subheader("""步驟四：啟動 SLAM""")
st.image("files/0031.png")
st.caption("啟動 SLAM")


st.subheader("""步驟五：掃描地形""")

col1, col2 = st.columns(2)
with col1:
    st.image("files/007.jpg")
    st.caption("掃描地形1")
with col2:
    st.image("files/0024.png")
    st.caption("掃描地形2")


st.subheader("""步驟六：路徑規劃""")
st.image("files/003.jpg")
st.caption("路徑規劃與導航展示圖")
st.subheader("""實際避障影片""")
st.video("files/001.mp4")
st.caption("TurtleBot 實際避障與導航演示影片")

