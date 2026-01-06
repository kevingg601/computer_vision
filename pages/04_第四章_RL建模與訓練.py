import streamlit as st
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time
import os
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 0. 頁面配置與視覺美化 (修正圖片重疊問題)
# ==========================================
st.set_page_config(page_title="第四章 - RL 建模與訓練", layout="wide")

st.title("🤖 第四章：強化學習 (RL) 建模與訓練實作")
st.markdown("---")

# 上方：文字說明區
st.info("### 📌 強化學習實驗流程說明")
st.markdown("""
本章節實作了三連桿機械臂（3-link Planar Manipulator）的智慧化控制。
透過正向運動學建立幾何模型後，在 Gymnasium 環境中定義強化學習的三大要素：

1. **狀態 (State)**：包含連桿角度、末端點位置與目標物座標。
2. **動作 (Action)**：各關節的旋轉角速度增量。
3. **獎勵 (Reward)**：引導末端點趨近目標球體之數學回饋機制。

**核心開發流程：**
`1.RL建模` → `2.算法配置` → `3.訓練迭代` → `4.成果驗證`
""")

# 下方：圖片並列區 (左：模型架構, 右：環境展示)
col_img1, col_img2 = st.columns(2)

with col_img1:
    if os.path.exists("files/04_model.png"):
        st.image("files/04_model.png", use_container_width=True)
        st.caption("<p style='text-align: center;'>圖 4-1：機械臂運動學鏈結模型 (Kinematic Chain)</p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ 找不到檔案：files/04_model.png")

with col_img2:
    if os.path.exists("files/04_demo.png"):
        st.image("files/04_demo.png", use_container_width=True)
        st.caption("<p style='text-align: center;'>圖 4-2：強化學習 Gymnasium 環境物理建模展示</p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ 找不到檔案：files/04_demo.png")

# 詳細程式碼說明 (針對 RL 核心檔案解說)
with st.expander("🔍 點擊查看：本章節核心程式檔 (env.py, rl.py, main.py) 深度技術解析", expanded=False):
    tab_code1, tab_code2, tab_code3 = st.tabs(["📄 env.py (物理環境)", "📄 rl.py (演算法核心)", "📄 main.py (訓練流程)"])
    
    with tab_code1:
        st.markdown("#### 1. 物理環境建模核心 (`env.py`) ")
        st.write("""
        本檔案負責機械臂的幾何邏輯與運動學計算：
        - **正向運動學 (FK)**：透過角度變量實時計算三節連桿末端的空間座標。
        - **狀態空間**：串接連桿位置與目標物距離數據，並進行歸一化處理。
        - **獎勵邏輯**：採負距離引導機制，當末端點接觸目標時賦予正向獎勵。
        """)
        
    with tab_code2:
        st.markdown("#### 2. 強化學習演算法實作 (`rl.py`) ")
        st.write("""
        定義了 DDPG / PPO 強化學習框架：
        - **Actor 網路**：學習如何根據當前座標輸出各關節的最佳旋轉量。
        - **Critic 網路**：作為評論者，評估動作的價值以優化策略。
        - **經驗回放**：打破數據時間相關性，提升訓練穩定度。
        """)

    with tab_code3:
        st.markdown("#### 3. 系統訓練循環 (`main.py`) ")
        st.write("""
        統籌整個學習過程：
        - **探索機制**：前期透過雜訊引導 Agent 進行嘗試，後期收斂至精確路徑。
        - **模型保存**：將訓練完成的權重保存為權重檔，供實作階段調用。
        """)

st.markdown("---")

# ==========================================
# 1. 共用工具與數據分析函式
# ==========================================

class TrainCallback(BaseCallback):
    def __init__(self, check_freq: int, plot_container):
        super(TrainCallback, self).__init__(verbose=1)
        self.check_freq = check_freq
        self.plot_container = plot_container
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                self.rewards.append(mean_reward)
                self.timesteps.append(self.num_timesteps)
                
                with self.plot_container:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=self.timesteps, y=self.rewards,
                        mode='lines+markers', name='Current',
                        line=dict(color='#00FF00', width=2)
                    ))
                    fig.update_layout(
                        title="Real-time Training Curve", xaxis_title="Steps", yaxis_title="Mean Reward",
                        template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        return True

def parse_urdf_lengths(filename):
    default_lengths = [0.4, 0.3, 0.2]
    search_paths = [filename, os.path.join("..", filename), os.path.join(os.path.dirname(__file__), filename), os.path.join(os.path.dirname(__file__), "..", filename)]
    target_path = None
    for path in search_paths:
        if os.path.exists(path): target_path = path; break
    if target_path is None: return default_lengths, "⚠️ 未偵測到檔案，使用預設參數。"
    try:
        tree = ET.parse(target_path)
        root = tree.getroot()
        lengths = []
        target_links = ['link1', 'link2', 'link3']
        for link_name in target_links:
            found = False
            for link in root.findall('link'):
                if link.get('name') == link_name:
                    box = link.find('.//visual/geometry/box')
                    if box is not None:
                        lengths.append(float(box.get('size').split()[0]))
                        found = True
                        break
            if not found: lengths.append(default_lengths[len(lengths)])
        return lengths, "✅ 成功讀取 URDF 參數！"
    except Exception as e: return default_lengths, f"❌ 讀取錯誤: {e}"

def plot_combined_history(history_list, title="Model Comparison"):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly 
    for i, rec in enumerate(history_list):
        short_name = f"Model {i+1}" 
        full_info = rec['label']
        color_idx = i % len(colors)
        fig.add_trace(go.Scatter(
            x=rec['steps'], y=rec['rewards'],
            mode='lines', 
            name=short_name,
            line=dict(width=2, color=colors[color_idx]),
            hovertemplate=f"<b>{short_name}</b><br>Step: %{{x}}<br>Reward: %{{y:.2f}}<br><i>{full_info}</i>"
        ))
    fig.update_layout(
        title=title, xaxis_title="Steps", yaxis_title="Reward",
        template="plotly_dark", height=400, 
        hovermode="x unified",
        legend_title_text='Models'
    )
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

def show_evaluation_metrics(history_list):
    st.subheader("📊 模型評分與排行 (Evaluation)")
    st.write("**評分標準：** 最終收斂獎勵 (Final Mean Reward)。")
    data = []
    for i, rec in enumerate(history_list):
        final_score = np.mean(rec['rewards'][-5:]) if len(rec['rewards']) > 0 else -999
        short_name = f"Model {i+1}"
        data.append({
            "ID": short_name, 
            "Score": final_score, 
            "Details": rec['label']
        })
    if not data: return
    df_score = pd.DataFrame(data).sort_values(by="Score", ascending=True) 
    fig = go.Figure(go.Bar(
        x=df_score["Score"],
        y=df_score["ID"],
        orientation='h',
        text=df_score["Score"].apply(lambda x: f"{x:.2f}"), 
        textposition='auto',
        marker=dict(color=df_score["Score"], colorscale='Viridis'),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}<br>%{customdata}",
        customdata=df_score["Details"]
    ))
    fig.update_layout(
        title="Model Performance Ranking",
        xaxis_title="Final Score",
        yaxis_title="Model ID",
        template="plotly_dark",
        height=300 + (len(data) * 30),
        margin=dict(l=100)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 2. 繪圖與動畫核心
# ==========================================

def create_animation(env, model, steps=150, is_3d=False, early_stop=False):
    obs, _ = env.reset()
    history_x, history_y, history_z = [], [], []
    history_ball = []
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        coords = env.get_coords() 
        history_x.append(coords[0]); history_y.append(coords[1]); history_z.append(coords[2])
        if hasattr(env, 'ball_pos'): history_ball.append(env.ball_pos.copy())
        else: history_ball.append(env.target)
        if early_stop and done:
            for _ in range(20):
                history_x.append(coords[0]); history_y.append(coords[1]); history_z.append(coords[2])
                if hasattr(env, 'ball_pos'): history_ball.append(env.ball_pos.copy())
                else: history_ball.append(env.target)
            break
    fig = go.Figure()
    ScatterClass = go.Scatter3d if is_3d else go.Scatter
    colors = ['red', '#00CC00', 'blue'] 
    for i in range(3):
        x_seg = [history_x[0][i], history_x[0][i+1]]
        y_seg = [history_y[0][i], history_y[0][i+1]]
        z_seg = [history_z[0][i], history_z[0][i+1]]
        trace_data = dict(x=x_seg, y=y_seg, mode='markers+lines', line=dict(color=colors[i], width=15 if is_3d else 8), marker=dict(size=8, color='black'), name=f'Link {i+1}')
        if is_3d: trace_data['z'] = z_seg
        fig.add_trace(ScatterClass(**trace_data))
    ball_init = history_ball[0]
    bx, by = ball_init[0], ball_init[1]
    bz = ball_init[2] if len(ball_init) > 2 else 0 
    ball_data = dict(x=[bx], y=[by], mode='markers', marker=dict(size=15, color='gold', symbol='circle', line=dict(width=2, color='black')), name='Yellow Ball')
    if is_3d: ball_data['z'] = [bz]
    fig.add_trace(ScatterClass(**ball_data))
    bx_init, by_init = history_x[0][0], history_y[0][0]
    base_data = dict(x=[bx_init], y=[by_init], mode='markers', marker=dict(size=20, color='#333', symbol='square'), name='Base')
    if is_3d: base_data['z'] = [0.05]
    fig.add_trace(ScatterClass(**base_data))
    if hasattr(env, 'box_pos'):
        box_pos = env.box_pos
        if is_3d:
            fig.add_trace(go.Mesh3d(x=[box_pos[0]-0.2, box_pos[0]+0.2, box_pos[0]+0.2, box_pos[0]-0.2], y=[box_pos[1]-0.2, box_pos[1]-0.2, box_pos[1]+0.2, box_pos[1]+0.2], z=[0.01]*4, color='green', opacity=0.3, name='Green Box'))
    frames = []
    for k in range(len(history_x)):
        frame_data = []
        for i in range(3):
            xs, ys, zs = [history_x[k][i], history_x[k][i+1]], [history_y[k][i], history_y[k][i+1]], [history_z[k][i], history_z[k][i+1]]
            if is_3d: frame_data.append(go.Scatter3d(x=xs, y=ys, z=zs))
            else: frame_data.append(go.Scatter(x=xs, y=ys))
        b_cur = history_ball[k]
        if is_3d: frame_data.append(go.Scatter3d(x=[b_cur[0]], y=[b_cur[1]], z=[b_cur[2] if len(b_cur)>2 else 0]))
        else: frame_data.append(go.Scatter(x=[b_cur[0]], y=[b_cur[1]]))
        bx_k, by_k = history_x[k][0], history_y[k][0]
        if is_3d: frame_data.append(go.Scatter3d(x=[bx_k], y=[by_k], z=[0.05]))
        else: frame_data.append(go.Scatter(x=[bx_k], y=[by_k]))
        if hasattr(env, 'box_pos') and is_3d:
            box_pos = env.box_pos
            frame_data.append(go.Mesh3d(x=[box_pos[0]-0.2, box_pos[0]+0.2, box_pos[0]+0.2, box_pos[0]-0.2], y=[box_pos[1]-0.2, box_pos[1]-0.2, box_pos[1]+0.2, box_pos[1]+0.2], z=[0.01]*4, color='green', opacity=0.3))
        frames.append(go.Frame(data=frame_data, name=str(k)))
    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(type="buttons", buttons=[dict(label="▶️ Play (Slow)", method="animate", args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True)])])],
        height=900, hovermode='closest', margin=dict(l=0, r=0, b=0, t=0), legend=dict(x=0, y=1, font=dict(color="white"))
    )
    limit = 2.5 if is_3d else 4.0
    if is_3d:
        fig.update_layout(scene=dict(xaxis=dict(range=[-4.0, 4.0], title='X'), yaxis=dict(range=[-4.0, 4.0], title='Y'), zaxis=dict(range=[0, 2.0], title='Z'), aspectratio=dict(x=1, y=1, z=0.5)), paper_bgcolor="rgba(0,0,0,0)")
    else:
        fig.update_layout(xaxis=dict(range=[-limit, limit]), yaxis=dict(range=[-2, 2]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ==========================================
# 3. 環境定義
# ==========================================

class EnvBasic2D(gym.Env):
    def __init__(self):
        super().__init__(); self.l = [0.4, 0.3, 0.2]
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.target = np.array([0.5, 0.5], dtype=np.float32); self.max_steps = 200; self.current_step = 0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.angles = np.random.uniform(-np.pi, np.pi, size=(3,))
        self.target = np.array([0.5, 0.5]); self.current_step = 0
        return self._get_obs(), {}
    def _get_obs(self):
        coords = self.get_coords(); ee = np.array([coords[0][-1], coords[1][-1]])
        return np.concatenate([self.angles, self.target, ee]).astype(np.float32)
    def get_coords(self):
        x, y = [0], [0]; cx, cy = 0, 0
        for i, length in enumerate(self.l):
            angle_sum = np.sum(self.angles[:i+1]); cx += length * np.cos(angle_sum); cy += length * np.sin(angle_sum); x.append(cx); y.append(cy)
        return x, y, [0]*4 
    def step(self, action):
        self.angles += action * 0.5; self.angles = np.arctan2(np.sin(self.angles), np.cos(self.angles)); self.current_step += 1
        coords = self.get_coords(); ee = np.array([coords[0][-1], coords[1][-1]])
        dist = np.linalg.norm(ee - self.target); reward = -(dist * 0.5) - 0.1 * np.linalg.norm(action)
        done = False; 
        if dist < 0.15: reward += 100; done = True
        if self.current_step >= self.max_steps: done = True
        return self._get_obs(), reward, done, False, {}

class EnvBasic3D(gym.Env):
    def __init__(self):
        super().__init__(); self.l = [0.4, 0.3, 0.2]
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.target = np.zeros(3); self.max_steps = 200; self.current_step = 0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.angles = np.random.uniform(-1.5, 1.5, size=(3,))
        self.target = np.array([0.5, 0.5, 0.1]); self.current_step = 0
        return self._get_obs(), {}
    def _get_obs(self):
        coords = self.get_coords(); ee = np.array([coords[0][-1], coords[1][-1], coords[2][-1]])
        return np.concatenate([self.angles, self.target, ee]).astype(np.float32)
    def get_coords(self):
        x, y, z = [0, 0], [0, 0], [0, self.l[0]]; ang = self.angles
        r2 = self.l[1] * np.cos(ang[1]); z2 = self.l[0] + self.l[1] * np.sin(ang[1]); x2 = r2 * np.cos(ang[0]); y2 = r2 * np.sin(ang[0])
        r3 = r2 + self.l[2] * np.cos(ang[1]+ang[2]); z3 = z2 + self.l[2] * np.sin(ang[1]+ang[2]); x3 = r3 * np.cos(ang[0]); y3 = r3 * np.sin(ang[0])
        x.extend([x2, x3]); y.extend([y2, y3]); z.extend([z2, z3]); return x, y, z
    def step(self, action):
        self.angles += action * 0.5; self.current_step += 1
        coords = self.get_coords(); ee = np.array([coords[0][-1], coords[1][-1], coords[2][-1]])
        dist = np.linalg.norm(ee - self.target); reward = -(dist * 0.5) - 0.1 * np.linalg.norm(action)
        done = False; 
        if dist < 0.15: reward += 100; done = True
        if self.current_step >= self.max_steps: done = True
        return self._get_obs(), reward, done, False, {}

class EnvAdvanced2D(gym.Env):
    def __init__(self, links):
        super().__init__(); self.l = links
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.target = np.array([1.0, 0.5], dtype=np.float32); self.max_steps = 200; self.current_step = 0; self.base_x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.base_x = 0.0; self.angles = np.random.uniform(-1.5, 1.5, size=(3,))
        self.target = np.array([1.5, 0.2]); self.current_step = 0
        return self._get_obs(), {}
    def _get_obs(self):
        coords = self.get_coords(); ee = np.array([coords[0][-1], coords[1][-1]])
        return np.concatenate([[self.base_x], self.angles, self.target, ee]).astype(np.float32)
    def get_coords(self):
        bx = self.base_x; by = 0.1; x, y = [bx], [by]; cx, cy = bx, by
        for i, length in enumerate(self.l):
            angle_sum = np.sum(self.angles[:i+1]); cx += length * np.cos(angle_sum); cy += length * np.sin(angle_sum); x.append(cx); y.append(cy)
        return x, y, [0]*4
    def step(self, action):
        self.base_x += action[0]; self.angles += action[1:] * 0.5; self.current_step += 1
        coords = self.get_coords(); ee = np.array([coords[0][-1], coords[1][-1]])
        dist = np.linalg.norm(ee - self.target); reward = -(dist * 0.5) - 0.1 * np.linalg.norm(action)
        done = False; 
        if dist < 0.15: reward += 100; done = True
        if self.current_step >= self.max_steps: done = True
        return self._get_obs(), reward, done, False, {}

class EnvAdvanced3D(gym.Env):
    def __init__(self, links):
        super().__init__(); self.l = links
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.target = np.zeros(3); self.max_steps = 200; self.current_step = 0; self.base_pos = np.zeros(2)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.base_pos = np.zeros(2); self.angles = np.random.uniform(-1.0, 1.0, size=(3,))
        self.target = np.array([np.random.uniform(1.5, 3.5) * np.random.choice([-1, 1]), np.random.uniform(1.5, 3.5) * np.random.choice([-1, 1]), 0.1]); self.current_step = 0
        return self._get_obs(), {}
    def _get_obs(self):
        coords = self.get_coords(); ee = np.array([coords[0][-1], coords[1][-1], coords[2][-1]])
        return np.concatenate([self.base_pos, self.angles, self.target, ee]).astype(np.float32)
    def get_coords(self):
        bx, by = self.base_pos; x, y, z = [bx, bx], [by, by], [0.1, self.l[0] + 0.1]; ang = self.angles
        r2 = self.l[1] * np.cos(ang[1]); z2 = z[-1] + self.l[1] * np.sin(ang[1]); x2 = bx + r2 * np.cos(ang[0]); y2 = by + r2 * np.sin(ang[0])
        r3 = r2 + self.l[2] * np.cos(ang[1]+ang[2]); z3 = z2 + self.l[2] * np.sin(ang[1]+ang[2]); x3 = bx + r3 * np.cos(ang[0]); y3 = by + r3 * np.sin(ang[0])
        x.extend([x2, x3]); y.extend([y2, y3]); z.extend([z2, z3]); return x, y, z
    def step(self, action):
        self.base_pos += action[:2] * 3.0; self.angles += action[2:] * 0.5; self.current_step += 1
        coords = self.get_coords(); ee = np.array([coords[0][-1], coords[1][-1], coords[2][-1]])
        dist = np.linalg.norm(ee - self.target); reward = -(dist * 0.5) - 0.01 * np.linalg.norm(action)
        done = False; 
        if dist < 0.2: reward += 100; done = True
        if self.current_step >= self.max_steps: done = True
        return self._get_obs(), reward, done, False, {}

class EnvFinalTask(EnvAdvanced3D):
    def __init__(self, links):
        super().__init__(links)
        self.ball_pos = np.zeros(3)
        self.box_pos = np.zeros(3)
        self.has_ball = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.has_ball = False
        r = np.random.uniform(1.2, 1.8)
        t = np.random.uniform(0, 2*np.pi)
        self.ball_pos = np.array([r*np.cos(t), r*np.sin(t), 0.1])
        self.box_pos = np.array([r*np.cos(t+np.pi), r*np.sin(t+np.pi), 0.1])
        self.target = self.ball_pos.copy() 
        return self._get_obs(), {}

    def step(self, action):
        # 攔截父類別的 done，不讓它因為碰到球就結束
        obs, reward, done_parent, trunc, info = super().step(action)
        done = False 
        
        coords = self.get_coords()
        ee = np.array([coords[0][-1], coords[1][-1], coords[2][-1]])
        
        if not self.has_ball:
            # 階段一：抓球
            if np.linalg.norm(ee - self.ball_pos) < 0.2:
                self.has_ball = True
                self.target = self.box_pos.copy() # 更新目標為方格
                reward += 100.0
        else:
            # 階段二：運球
            self.ball_pos = ee.copy()
            if np.linalg.norm(ee - self.box_pos) < 0.2:
                reward += 200.0
                done = True # 放入箱子才真正 Done
        
        if self.current_step >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, trunc, info

# ==========================================
# 4. Streamlit 主控制介面
# ==========================================

# 數學定義 Expander
with st.expander("📘 點擊查看：RL 數學建模與程式架構詳解", expanded=False):
    st.markdown("### 1. 系統座標變換 (Kinematics)")
    st.latex(r'''x = x_{base} + \sum l_i \cos(\sum \theta_i), \quad y = y_{base} + \sum l_i \sin(\sum \theta_i)''')
    st.markdown("### 2. 核心類別說明")
    st.write("""
    - **`gym.Env` 類別 (如 EnvBasic2D)**: 繼承 Gymnasium 框架，內部實作了機械臂正向運動學與負距離獎勵邏輯。
    - **`TrainCallback`**: 繼承 SB3 類別，用於在訓練過程中即時擷取平均獎勵並反饋至 Plotly 圖表。
    - **`create_animation`**: 將訓練好的神經網路推理過程紀錄，轉換為 Plotly 動畫幀進行可視化呈現。
    """)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "2D 基礎訓練", "3D 基礎訓練", "2D 協同控制", 
    "3D 全向移動", "🏆 最終成果展示"
])

# 會話狀態初始化
if 'h1' not in st.session_state: st.session_state.h1 = []
if 'h2' not in st.session_state: st.session_state.h2 = []
if 'h3' not in st.session_state: st.session_state.h3 = []
if 'h4' not in st.session_state: st.session_state.h4 = []

# 訓練核心函式
def run_tab(tab_obj, env_class, history_key, env_args=None, label="Model", is_3d=False):
    with tab_obj:
        st.subheader(f"🛠️ {label} 訓練核心")
        c1, c2 = st.columns([1, 2])
        with c1:
            steps = st.slider(f"訓練步數", 2000, 50000, 5000, key=f"s_{label}")
            lr = st.select_slider(f"學習率 ({label})", [0.0001, 0.0003, 0.001], value=0.0003, key=f"l_{label}")
            if st.button(f"🚀 開始強化學習訓練", key=f"b_{label}"):
                env = env_class(env_args) if env_args else env_class()
                model = PPO("MlpPolicy", env, verbose=0, learning_rate=lr, device="cpu")
                with c2: 
                    ph = st.empty(); cb = TrainCallback(1000, ph)
                    with st.spinner("權重優化中..."): model.learn(total_timesteps=steps, callback=cb)
                    getattr(st.session_state, history_key).append({"label": f"{label}-{datetime.now().strftime('%H:%M')}", "model": model, "steps": cb.timesteps, "rewards": cb.rewards})

        hist = getattr(st.session_state, history_key)
        if hist:
            show_evaluation_metrics(hist); plot_combined_history(hist)
            if st.button("▶️ 執行離線模擬動畫", key=f"sim_{label}"):
                env = env_class(env_args) if env_args else env_class()
                fig = create_animation(env, hist[-1]['model'], steps=150, is_3d=is_3d)
                st.plotly_chart(fig, use_container_width=True)

# 啟動 Tab 1~4
urdf_len, _ = parse_urdf_lengths("robot_arm.urdf")
run_tab(tab1, EnvBasic2D, 'h1', label="Basic2D", is_3d=False)
run_tab(tab2, EnvBasic3D, 'h2', label="Basic3D", is_3d=True)
run_tab(tab3, EnvAdvanced2D, 'h3', env_args=urdf_len, label="Advanced2D", is_3d=False)
run_tab(tab4, EnvAdvanced3D, 'h4', env_args=urdf_len, label="Advanced3D", is_3d=True)

# --- Tab 5: 雙成果展示 (影片成果 + 模型模擬) ---
with tab5:
    st.header("🏆 智慧機械整合：最終實作成果展示")
    
    # 成果展示一：線上UI實作
    st.subheader("1. 最終任務實作成果 (Pick & Place)")
    cv1, cv2 = st.columns([1.6, 1])
    with cv1:
        if os.path.exists("files/04_final.mp4"):
            st.video("files/04_final.mp4")
            st.caption("影片一：Agent 載入 Tab 4 權重後，執行連續抓取放置任務展示。")
    with cv2:
        st.success("**實作邏輯說明：**\n整合移動底座與機械臂策略。Agent 需先移動至目標球體範圍，完成耦合後將目標運送至綠色箱子上方完成任務。")
        
    st.divider()

    # 成果展示二：連桿追隨鼠標
    st.subheader("2. 動態追蹤實作成果 (追逐鼠標實作)")
    cv3, cv4 = st.columns([1.6, 1])
    with cv3:
        if os.path.exists("files/04_demo.mp4"):
            st.video("files/04_demo.mp4")
            st.caption("影片二：展示 Agent 即時捕捉滑鼠目標座標，並主動計算關節角度進行追逐。")
    with cv4:
        st.success("**實作邏輯說明：**\n此功能展現了神經網路權重的泛化能力。連桿底座固定，僅透過讀取 Tab 1 訓練好的『大腦』來解算動態目標點位置。")

    st.divider()
    # 保留模型選擇與模擬按鈕
    st.subheader("🎬 現有訓練模型之離線實作驗證")
    if len(st.session_state.h4) > 0:
        opts_f = [f"Model {i+1}: {h['label']}" for i, h in enumerate(st.session_state.h4)]
        sel_idx_f = st.selectbox("🎯 選擇要實作的訓練模型版本", range(len(opts_f)), format_func=lambda x: opts_f[x], key="sel_final")
        if st.button("▶️ 啟動實作 Plotly 動畫", key="sim_final"):
            env = EnvFinalTask(urdf_len)
            fig_anim = create_animation(env, st.session_state.h4[sel_idx_f]['model'], steps=200, is_3d=True, early_stop=True)
            st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.info("💡 提示：在 Tab 4 完成任意訓練後，此處將顯示模型列表供模擬實作。")