"""
批量为 Streamlit 页面添加图片 caption
"""
import os
import re

# 定义文件路径和对应的 caption 映射
pages_dir = r"e:\Salamander\final_2\IMD_final\computer_vision\pages"

# 读取文件05
file05_path = os.path.join(pages_dir, "05_第五章_分工自評表與心得.py")

with open(file05_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 添加 caption
if 'st.image("files/0032.jpg")' in content and 'st.caption' not in content:
    content = content.replace(
        'st.image("files/0032.jpg")',
        'st.image("files/0032.jpg")\nst.caption("本組分工自評表")'
    )
    
    with open(file05_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 已為第五章添加 caption")
else:
    print("⚠️ 第五章已有 caption 或文件内容不匹配")

# 读取文件01  
file01_path = os.path.join(pages_dir, "01_第一章 電腦視覺概論.py")

with open(file01_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 添加缺失的 caption
if 'st.image("files/0030.png")' in content:
    # 检查下一行是否已经有 caption
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'st.image("files/0030.png")' in line:
            if i + 1 < len(lines) and 'st.caption' not in lines[i + 1]:
                lines.insert(i + 1, 'st.caption("Turtlebot3 安裝指令與執行結果")')
                break
    
    content = '\n'.join(lines)
    
    with open(file01_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 已為第一章添加缺失的 caption")
else:
    print("⚠️ 第一章已有所有 caption 或文件内容不匹配")

print("\n完成！")
