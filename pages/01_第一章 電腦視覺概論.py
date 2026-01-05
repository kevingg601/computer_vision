import streamlit as st
import pandas as pd
import numpy as np
# import openpyxl
import xlrd

import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode


st.title("第一章 電腦視覺概論")

st.write("電腦視覺（Computer vision）是一門研究如何使機器「看」的科學，更進一步的說，就是指用攝影機和電腦代替人眼對目標進行辨識、跟蹤和測量等機器視覺，並進一步做圖像處理，用電腦處理成為更適合人眼觀察或傳送給儀器檢測的圖像[1]。作為一門科學學科，電腦視覺研究相關的理論和技術，試圖建立能夠從圖像或者多維資料中取得「資訊」的人工智慧系統。這裡所指的資訊指香農定義的，可以用來幫助做一個「決定」的資訊。因為感知可以看作是從感官訊號中提取資訊，所以電腦視覺也可以看作是研究如何使人工系統從圖像或多維資料中「感知」的科學。")

# ========== [start] 以下為實驗報告基本架構 ==============

# 在這裡添加實驗一的具體內容，如圖表、數據等
st.header("1.1 電腦視覺概論")
st.write("將此實驗之目的以文字結合圖片之方式描述於此!")
# Displaying Plain Text
st.text("Hi,\nPeople\t!!!!!!!!!")
st.text('Welcome to')
st.text(""" Streamlit's World""")
st.text('text輸出會由新的一行開始!')

st.header("1.2 電腦視覺之產業應用")
st.write("將此實驗之原理以文字結合圖片之方式描述於此!")


st.header("實驗步驟：")
st.write("將此實驗之步驟以文字結合圖片之方式描述於此!")
# sub-header
st.subheader("""步驟一：""")
st.write("步驟一之說明描述")
st.subheader("""步驟二：""")
st.write("步驟二之說明描述")
st.subheader("""步驟三：""")
st.write("步驟三之說明描述")

st.header("實驗結果：")
st.write("將此實驗之結果以文字結合圖片之方式描述於此!")

st.header("實驗結論：")
st.write("將此實驗之結論以文字結合圖片之方式描述於此!")

st.header("參考文獻：")
st.write("將此實驗之參考文獻以文字之方式描述於此!")

# ========== [end] 以上為實驗報告基本架構 ==============

# streamlit好像沒有調整間距之功能，暫時以下列方式來調整!
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
# 
st.header("表格貼法：")
st.write("如果有實驗數據要匯入時，可複製下列程式段來修改!")
st.write("記得要匯入pandas, numpy, xlrd等套件!")
# defining random values in a dataframe using pandas and numpy
df = pd.DataFrame(
 np.random.randn(30, 10),
 columns=('col_no %d' % i for i in range(10)))
st.dataframe(df)
##
# Bar Chart
st.line_chart(df)


# 讀入已有之excel檔案資料之方法
# 參考資料[1]：https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
# 參考資料[2]：https://www.learncodewithmike.com/2020/12/read-excel-file-using-pandas.html
# xls_data = pd.read_excel(open('files/實驗1.雷諾數實驗(實驗數據).xlsx', 'rb'), sheet_name='Sheet1') 
xls_data = pd.read_excel('files/Exp1.xls') 
df = pd.DataFrame(xls_data)
st.dataframe(df)
##
# Bar Chart
df2 = pd.DataFrame(xls_data, columns=["測量水量", "雷諾數"])
st.bar_chart(df2)
# Line Chart
df2 = pd.DataFrame(xls_data, columns=["流速", "雷諾數"])
st.line_chart(df2)

#
st.header("影片貼法：")
st.write("如果有實驗影片要匯入時，可複製下列程式段來修改!")
# Open Video using filepath with filename and read the video file
sample_video = open("files/red_rock.mov", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 10)
# Caption
st.caption("""(This is Red Rock!)記得在影片或是照片後加上圖例說明!""")


# Displaying Python Code
st.write("如果有程式要插入，可以複製下列程式段來修改!")
st.subheader("""Python Code""")
code = '''def hello():
 print("Hello, Streamlit!")'''
st.code(code, language='python')



# ==================================================================

# 利用 Streamlit 顯示即時攝影機影像的步驟與範例 Python 程式碼：
# 記得安裝opencv-python, streamlit_webrtc套件
def main():
    st.title("即時 Webcam 影像顯示")
    st.write("本應用展示如何在 Streamlit 中顯示即時攝影機影像。")
    
    # 方法一：啟動攝影機
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV)
    
    # 方法二：啟動 OpenCV 攝影機擷取
    cap = cv2.VideoCapture(0)
    
    stframe = st.empty()  # 用於動態顯示影像的占位符
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("無法讀取攝影機影像")
            break
        
        # OpenCV 的 BGR 格式轉換為 RGB 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用 Streamlit 顯示影像
        stframe.image(frame, channels="RGB", use_column_width=True)
    
    # 釋放攝影機資源
    cap.release()

if __name__ == "__main__":
    main()
