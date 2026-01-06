# 为第一章添加缺失的 caption
import os

file_path = r"e:\Salamander\final_2\IMD_final\computer_vision\pages\01_第一章 電腦視覺概論.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找并插入 caption
for i, line in enumerate(lines):
    if 'files/0030.png' in line:
        # 检查下一行是否已经有 caption
        if i + 1 < len(lines) and 'st.caption' not in lines[i + 1]:
            lines.insert(i + 1, 'st.caption("Turtlebot3 安裝指令與執行結果")\n')
            print(f"✅ 在第 {i+2} 行插入 caption")
            break
        else:
            print("⚠️ Caption 已存在")
            break
else:
    print("⚠️ 未找到目标行")

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("完成！")
