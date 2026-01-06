import streamlit as st
import os

# --- 頁面配置 ---
st.set_page_config(page_title="第二章 - 深度學習影像移動控制", layout="wide")

st.title("🧠 第二章：深度學習影像移動控制實作")
st.markdown("""
本章節旨在實作基於深度學習之 AMR (Autonomous Mobile Robot) 影像控制系統。
透過實體機進行即時影像辨識，並將運動指令傳輸至虛擬機環境，驅動三連桿自走車執行相對應動作。
""")
st.markdown("---")

# --- 1. 視覺指令定義 ---
st.header("1. 視覺指令定義與編碼規範")
st.write("本系統採用標準化圖卡作為控制輸入，透過高對比度特徵強化 AI 模型之辨識率與系統穩定性。")

c1, c2, c3, c4, c5 = st.columns(5)
# 確保圖片放在 files/ 資料夾下，檔名與下方一致
with c1: st.image("files/forward.jpg", caption="前進指令 (Forward)")
with c2: st.image("files/backward.jpg", caption="後退指令 (Backward)")
with c3: st.image("files/left.jpg", caption="左轉指令 (Left)")
with c4: st.image("files/right.jpg", caption="右轉指令 (Right)")
with c5: st.image("files/stop.jpg", caption="停止指令 (Stop)")

st.markdown("""
| 指令標籤 | 運動學邏輯 | 控制 Topic (Message Type) |
| :--- | :--- | :--- |
| **Forward** | 依照當前角度 $\\theta$ 進行線速度位移 | `/cmd_vel` (geometry_msgs/Twist) |
| **Backward** | 依照當前角度 $\\theta$ 進行反向線速度位移 | `/cmd_vel` (geometry_msgs/Twist) |
| **Left** | 原地增加航向角變數，實現左轉自轉 | `/cmd_vel` (geometry_msgs/Twist) |
| **Right** | 原地減少航向角變數，實現右轉自轉 | `/cmd_vel` (geometry_msgs/Twist) |
| **Stop** | 線速度與角速度立即歸零 | `/cmd_vel` (geometry_msgs/Twist) |
""")

# --- 2. 模型訓練與評估數據 ---
st.header("2. 模型訓練成果與數據分析")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 訓練歷程分析")
    if os.path.exists("files/results.png"):
        st.image("files/results.png", use_container_width=True)
    st.write("模型採用 YOLOv8 分類架構進行 20 輪訓練，觀察曲線可見 Loss 持續下降並於第 15 輪後趨於穩定。")

with col2:
    st.subheader("📊 混淆矩陣 (Confusion Matrix)")
    if os.path.exists("files/confusion_matrix.png"):
        st.image("files/confusion_matrix.png", use_container_width=True)
    st.write("混淆矩陣顯示模型在各類別均具備極高準確率，對於左右轉向之辨識已透過信心門檻機制優化。")

# --- 3. 機器人建模與整合測試 ---
st.header("3. 機器人建模與系統模擬測試")
col_img, col_vid = st.columns([1, 1.5])

with col_img:
    st.subheader("🏗️ 數位雙生模型 (RViz)")
    if os.path.exists("files/rviz_model.png"):
        st.image("files/rviz_model.png", caption="三連桿自走車整合模型")
    st.markdown("""
    **模型架構說明：**
    - **底盤 (Base Link)**：定義兩輪差速移動平台之幾何尺寸。
    - **機械臂 (Manipulator)**：紅、綠、藍三連桿平面式架構，具備三個旋轉自由度。
    - **座標系 (TF Tree)**：建立 `odom` 到 `base_link` 之動態變換。
    """)

with col_vid:
    st.subheader("🎬 系統運作展示")
    if os.path.exists("files/final_demo.mp4"):
        st.video("files/final_demo.mp4")
        st.caption("Windows AI 辨識連動虛擬機 RViz 移動同步展示影片")

# --- 4. 系統程式源碼與邏輯解析 ---
st.header("4. 系統程式源碼解析")
st.write("本系統包含影像辨識、通訊轉接與運動驅動三大模組。")

tab1, tab2, tab3 = st.tabs(["🐍 視覺辨識與控制 (Python)", "🏗️ 機器人建模 (URDF/Xacro)", "🛰️ 系統配置 (Launch/RViz)"])

with tab1:
    with st.expander("📄 ai_robot_driver.py - 智慧控制核心"):
        st.write("負責從 Webcam 擷取影像並使用 YOLOv8 模型進行推理，再將結果轉換為 ROS 座標變換。")
        st.code("""
# 核心邏輯：AI 推理與 TF 廣播
results = model(frame, imgsz=224)
label = results[0].names[results[0].probs.top1].lower()
conf = result.probs.top1conf.item()

if conf > 0.4:
    if 'forward' in label:
        x += speed * cos(th)
        y += speed * sin(th)
    elif 'left' in label:
        th += turn_step
    # 發布座標變換至 RViz
    br.sendTransform((x, y, 0), quaternion, rospy.Time.now(), "base_link", "odom")
        """, language="python")

    with st.expander("📄 fake_driver.py - 運動學模擬測試"):
        st.write("提供基於鍵盤控制的平滑運動模型，用於驗證機器人本體運動學之正確性。")
        st.code("x += v * math.cos(th) * dt; y += v * math.sin(th) * dt; th += w * dt", language="python")

with tab2:
    with st.expander("📄 mobile_manipulator.urdf.xacro - 總體建模檔"):
        st.write("整合底盤與機械手臂模型，定義模組化連結關係。")
        st.code("""
<xacro:include filename="mobile_base.urdf.xacro" />
<xacro:include filename="my_manipulator.urdf.xacro" />
<joint name="arm_to_base" type="fixed">
    <parent link="base_link"/><child link="arm_base_link"/>
</joint>""", language="xml")

    with st.expander("📄 my_manipulator.urdf.xacro - 機械臂細節"):
        st.write("定義三連桿手臂各節長度、色彩與旋轉關節限位。")
        st.code("""
<joint name="joint1" type="revolute">
    <parent link="arm_base_link"/><child link="link1"/>
    <axis xyz="0 0 1"/><limit effort="1000" lower="-3.14" upper="3.14" velocity="0.5"/>
</joint>""", language="xml")

with tab3:
    with st.expander("📄 display.launch - 系統啟動配置"):
        st.write("自動加載機器人描述、啟動狀態發布器與 RViz 可視化環境。")
        st.code('<node name="rviz" pkg="rviz" type="rviz" args="-d $(arg rvizconfig)" />', language="xml")

    with st.expander("📄 41123128.rviz - 視角配置存檔"):
        st.write("保存 RViz 顯示設定，包括 Fixed Frame 設為 odom 與模型渲染參數。")

# --- 5. 操作指令手冊 ---
st.header("5. 系統操作指令手冊")
st.info("執行本系統時，請依照下列順序於各平台啟動對應程式：")

col_cli1, col_cli2 = st.columns(2)
with col_cli1:
    st.subheader("🖥️ 虛擬機 (Ubuntu)")
    st.code("""
# 1. 啟動 RViz 模擬器
roslaunch display.launch

# 2. 啟動跨平台通訊服務
roslaunch rosbridge_server rosbridge_websocket.launch

# 3. 啟動 AI 指令監聽與驅動程式
python3 ai_robot_driver.py
    """, language="bash")

with col_cli2:
    st.subheader("💻 實體機 (Windows)")
    st.code("""
# 1. 進入虛擬環境
.\\venv\\Scripts\\activate

# 2. 啟動 Streamlit 整合介面
streamlit run template.py
    """, language="bash")