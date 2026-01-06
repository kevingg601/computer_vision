import streamlit as st
import os

# --- 頁面配置 ---
st.set_page_config(page_title="第二章 - 深度學習影像移動控制", layout="wide")

# 初始化 session state 用於切換頁面模式
if 'page_mode' not in st.session_state:
    st.session_state.page_mode = 'virtual'

st.title("🧠 第二章：深度學習影像移動控制實作")
st.markdown("本系統展示從 **虛擬模擬 (Simulation)** 到 **實體部署 (Real World)** 的完整開發流程。")

# ==========================================
#      頂部導航欄 (大按鈕切換)
# ==========================================
st.markdown("### 🔽 請選擇展示模式")
btn_col1, btn_col2 = st.columns(2)

def get_btn_type(current_mode, target_mode):
    return "primary" if current_mode == target_mode else "secondary"

with btn_col1:
    if st.button("🖥️ 第一階段：虛擬環境模擬 (Virtual)", 
                 use_container_width=True, 
                 type=get_btn_type(st.session_state.page_mode, 'virtual')):
        st.session_state.page_mode = 'virtual'

with btn_col2:
    if st.button("🤖 第二階段：實體機器人實作 (Real)", 
                 use_container_width=True, 
                 type=get_btn_type(st.session_state.page_mode, 'real')):
        st.session_state.page_mode = 'real'

st.markdown("---")

# ==========================================
#      模式 A: 虛擬環境模擬 (已補回所有原始資料)
# ==========================================
if st.session_state.page_mode == 'virtual':
    st.header("🖥️ 第一階段：虛擬環境模擬")
    st.info("此部分展示基於 YOLO 與 RViz 的模擬驗證成果。")
    
    # --- 1. 視覺指令定義 ---
    st.subheader("1. 視覺指令定義與編碼規範")
    st.write("本系統採用標準化圖卡作為控制輸入，透過高對比度特徵強化 AI 模型之辨識率與系統穩定性。")

    c1, c2, c3, c4, c5 = st.columns(5)
    # [BUG FIX] 修正 DeltaGenerator 顯示問題
    with c1: 
        if os.path.exists("files/forward.jpg"): st.image("files/forward.jpg", caption="前進指令 (Forward)")
        else: st.info("缺檔: forward.jpg")
    with c2: 
        if os.path.exists("files/backward.jpg"): st.image("files/backward.jpg", caption="後退指令 (Backward)")
        else: st.info("缺檔: backward.jpg")
    with c3: 
        if os.path.exists("files/left.jpg"): st.image("files/left.jpg", caption="左轉指令 (Left)")
        else: st.info("缺檔: left.jpg")
    with c4: 
        if os.path.exists("files/right.jpg"): st.image("files/right.jpg", caption="右轉指令 (Right)")
        else: st.info("缺檔: right.jpg")
    with c5: 
        if os.path.exists("files/stop.jpg"): st.image("files/stop.jpg", caption="停止指令 (Stop)")
        else: st.info("缺檔: stop.jpg")

    st.markdown("""
    | 指令標籤 | 運動學邏輯 | 控制 Topic (Message Type) |
    | :--- | :--- | :--- |
    | **Forward** | 依照當前角度 $\\theta$ 進行線速度位移 | `/cmd_vel` (geometry_msgs/Twist) |
    | **Backward** | 依照當前角度 $\\theta$ 進行反向線速度位移 | `/cmd_vel` (geometry_msgs/Twist) |
    | **Left** | 原地增加航向角變數，實現左轉自轉 | `/cmd_vel` (geometry_msgs/Twist) |
    | **Right** | 原地減少航向角變數，實現右轉自轉 | `/cmd_vel` (geometry_msgs/Twist) |
    | **Stop** | 線速度與角速度立即歸零 | `/cmd_vel` (geometry_msgs/Twist) |
    """)

    # --- 2. 模型訓練 ---
    st.subheader("2. 模型訓練成果與數據分析")
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

    # --- 3. 機器人建模 ---
    st.subheader("3. 機器人建模與系統模擬測試")
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

    # --- 4. 程式碼解析 (原本的完整內容) ---
    st.header("4. 系統程式源碼解析")
    st.write("本系統包含影像辨識、通訊轉接與運動驅動三大模組。")

    # 這裡恢復了原本的三個分頁
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

    # --- 5. 操作手冊 (原本的完整內容) ---
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


# ==========================================
#      模式 B: 實體機器人實作 (新增內容)
# ==========================================
elif st.session_state.page_mode == 'real':
    st.header("🤖 第二階段：實體機器人實作")
    st.success("本階段說明如何將訓練好的模型部署至 Ubuntu 機器人環境中。")

    # --- 1. 標準圖示 ---
    st.subheader("1. 實體控制標準圖示")
    ic1, ic2, ic3, ic4, ic5 = st.columns(5)
    
    with ic1: 
        if os.path.exists("files/icon_forward.jpg"): st.image("files/icon_forward.jpg", caption="標準前進")
        else: st.info("缺檔: icon_forward.jpg")
    with ic2: 
        if os.path.exists("files/icon_backward.jpg"): st.image("files/icon_backward.jpg", caption="標準後退")
        else: st.info("缺檔: icon_backward.jpg")
    with ic3: 
        if os.path.exists("files/icon_left.jpg"): st.image("files/icon_left.jpg", caption="標準左轉")
        else: st.info("缺檔: icon_left.jpg")
    with ic4: 
        if os.path.exists("files/icon_right.jpg"): st.image("files/icon_right.jpg", caption="標準右轉")
        else: st.info("缺檔: icon_right.jpg")
    with ic5: 
        if os.path.exists("files/icon_stop.jpg"): st.image("files/icon_stop.jpg", caption="標準停止")
        else: st.info("缺檔: icon_stop.jpg")

    # --- 2. Teachable Machine 訓練 ---
    st.subheader("2. Teachable Machine 模型訓練")
    tm1, tm2 = st.columns(2)
    with tm1:
        st.write("📸 **訓練介面**")
        if os.path.exists("files/tm_training.jpg"):
            st.image("files/tm_training.jpg", caption="資料收集與訓練")
        else: st.warning("請放入 tm_training.jpg")
    with tm2:
        st.write("💾 **模型匯出**")
        if os.path.exists("files/tm_export.jpg"):
            st.image("files/tm_export.jpg", caption="匯出 Keras .h5 檔")
        else: st.warning("請放入 tm_export.jpg")

    # --- [NEW] 部署步驟說明 ---
    st.markdown("---")
    st.subheader("📖 詳細部署步驟指南 (Deployment Guide)")
    st.info("由於 Teachable Machine 是在雲端訓練，我們需要將模型下載並轉移至機器人控制器 (Ubuntu)。")

    step1, step2, step3 = st.columns(3)
    
    with step1:
        st.markdown("#### Step 1: 檔案準備")
        st.write("從 Teachable Machine 下載 ZIP 檔後解壓縮，會得到兩個關鍵檔案：")
        st.code("""
1. keras_model.h5 (權重檔)
2. labels.txt (類別標籤)
        """, language="text")
        st.write("請將這兩個檔案放入 Ubuntu 專案資料夾中。")

    with step2:
        st.markdown("#### Step 2: 環境安裝")
        st.write("在 Ubuntu 終端機執行以下指令，安裝 Tensorflow 與 OpenCV：")
        st.code("""
# 更新 pip
pip3 install --upgrade pip

# 安裝必要套件
pip3 install tensorflow
pip3 install opencv-python
pip3 install rospkg
        """, language="bash")

    with step3:
        st.markdown("#### Step 3: 啟動控制")
        st.write("確認 TurtleBot3 底層已啟動後，執行我們的 AI 驅動程式：")
        st.code("""
# 1. 啟動機器人底層
roslaunch turtlebot3_bringup turtlebot3_robot.launch

# 2. 啟動 AI 控制腳本
python3 real_robot_driver.py
        """, language="bash")

    # --- 3. 實體操作展示 (Tabs) ---
    st.markdown("---")
    st.subheader("3. 實體機操作展示 (Sim-to-Real)")
    
    tab_f, tab_b, tab_l, tab_r, tab_s = st.tabs([
        "⬆️ 前進", "⬇️ 後退", "⬅️ 左轉", "➡️ 右轉", "🛑 停止"
    ])

    # 前進 Tab
    with tab_f:
        c_view, c_video = st.columns([1, 2])
        with c_view:
            st.markdown("#### 🤖 機器人視野")
            if os.path.exists("files/view_forward.jpg"): st.image("files/view_forward.jpg")
            else: st.warning("缺檔: view_forward.jpg")
        with c_video:
            st.markdown("#### 🎬 實測影片")
            if os.path.exists("files/video_forward.mp4"): st.video("files/video_forward.mp4")
            else: st.info("缺檔: video_forward.mp4")

    # 後退 Tab
    with tab_b:
        c_view, c_video = st.columns([1, 2])
        with c_view:
            st.markdown("#### 🤖 機器人視野")
            if os.path.exists("files/view_backward.jpg"): st.image("files/view_backward.jpg")
            else: st.warning("缺檔: view_backward.jpg")
        with c_video:
            st.markdown("#### 🎬 實測影片")
            if os.path.exists("files/video_backward.mp4"): st.video("files/video_backward.mp4")
            else: st.info("缺檔: video_backward.mp4")

    # 左轉 Tab
    with tab_l:
        c_view, c_video = st.columns([1, 2])
        with c_view:
            st.markdown("#### 🤖 機器人視野")
            if os.path.exists("files/view_left.jpg"): st.image("files/view_left.jpg")
            else: st.warning("缺檔: view_left.jpg")
        with c_video:
            st.markdown("#### 🎬 實測影片")
            if os.path.exists("files/video_left.mp4"): st.video("files/video_left.mp4")
            else: st.info("缺檔: video_left.mp4")

    # 右轉 Tab
    with tab_r:
        c_view, c_video = st.columns([1, 2])
        with c_view:
            st.markdown("#### 🤖 機器人視野")
            if os.path.exists("files/view_right.jpg"): st.image("files/view_right.jpg")
            else: st.warning("缺檔: view_right.jpg")
        with c_video:
            st.markdown("#### 🎬 實測影片")
            if os.path.exists("files/video_right.mp4"): st.video("files/video_right.mp4")
            else: st.info("缺檔: video_right.mp4")

    # 停止 Tab
    with tab_s:
        c_view, c_video = st.columns([1, 2])
        with c_view:
            st.markdown("#### 🤖 機器人視野")
            if os.path.exists("files/view_stop.jpg"): st.image("files/view_stop.jpg")
            else: st.warning("缺檔: view_stop.jpg")
        with c_video:
            st.markdown("#### 🎬 實測影片")
            if os.path.exists("files/video_stop.mp4"): st.video("files/video_stop.mp4")
            else: st.info("缺檔: video_stop.mp4")

    # --- 4. 實體程式碼解析 ---
    st.markdown("---")
    st.subheader("4. 實體控制核心程式碼")
    with st.expander("📄 real_robot_driver.py - 整合 Keras 與 ROS Twist"):
        st.code("""
from keras.models import load_model
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist

# 初始化 ROS Node
rospy.init_node('ai_driver')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

# 載入 Teachable Machine 匯出的模型
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
camera = cv2.VideoCapture(0)

while not rospy.is_shutdown():
    ret, image = camera.read()
    # 預處理圖片
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1

    # 推論
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    action = class_names[index].strip()
    
    # 控制邏輯
    twist = Twist()
    if "Forward" in action:
        twist.linear.x = 0.1
    elif "Backward" in action:
        twist.linear.x = -0.1
    elif "Left" in action:
        twist.angular.z = 0.5
    elif "Right" in action:
        twist.angular.z = -0.5
    else:
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        
    pub.publish(twist)
        """, language="python")