"""
================================================================================
超智通 - 超来科技智能知识库系统
Web应用入口文件 (app.py)

【作用】
这个文件是整个系统的"大门"，负责：
1. 接收用户的网页请求
2. 处理文档上传
3. 处理用户提问
4. 返回网页给浏览器

【Flask是什么】
Flask是一个用Python写的Web框架，可以帮助我们快速搭建网站。
就像盖房子，Flask提供了地基和框架，我们只需要装修（写业务逻辑）就行。

【如何运行】
在命令行输入: python app.py
然后打开浏览器访问: http://127.0.0.1:5000
================================================================================
"""

# 导入需要的工具
# Flask: Web框架的核心
# render_template: 渲染HTML页面
# request: 获取用户请求数据
# flash: 显示提示信息
# redirect: 页面跳转
# jsonify: 返回JSON数据
from flask import Flask, render_template, request, flash, redirect, jsonify
import os
import re

# 从main.py导入所有功能
# main.py包含了RAG的核心逻辑（文档处理、向量检索、AI回答）
from main import *

# =============================================================================
# 第一步：创建Flask应用对象
# =============================================================================
# Flask(__name__) 创建一个Flask应用实例
# __name__ 是Python的特殊变量，表示当前模块的名字
# template_folder默认app同级目录下名为templates的目录
app = Flask(__name__, template_folder='web_templates')


# =============================================================================
# 第二步：配置文件上传功能
# =============================================================================
# UPLOAD_FOLDER: 上传的文件保存在哪个文件夹
# 这里设置为 'uploads'，表示文件会保存在项目目录下的uploads文件夹中
UPLOAD_FOLDER = '../uploads'

# 检查uploads文件夹是否存在，如果不存在就创建它
# os.path.exists() 检查路径是否存在
# os.makedirs() 创建文件夹
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 将上传文件夹的配置告诉Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# =============================================================================
# 第三步：设置允许上传的文件类型
# =============================================================================
# ALLOWED_EXTENSIONS 是一个集合（set），存储允许的文件扩展名
# 目前只允许上传 .docx (Word文档) 和 .pdf 文件
ALLOWED_EXTENSIONS = {'docx', 'pdf'}

def allowed_file(filename):
    """
    【函数作用】检查文件是否允许上传
    
    【参数】
        filename: 文件名，比如 "合同.docx"
    
    【返回值】
        True: 文件类型允许上传
        False: 文件类型不允许
    
    【判断逻辑】
        1. 文件名中必须有 '.' (点号)
        2. 点号后面的部分（扩展名）必须在 ALLOWED_EXTENSIONS 中
    
    【例子】
        allowed_file("合同.docx") -> True
        allowed_file("照片.jpg") -> False
    """
    # 检查文件名中是否有'.'
    has_dot = '.' in filename
    
    # rsplit('.', 1) 从右边开始分割，最多分割1次
    # "合同.docx".rsplit('.', 1) -> ["合同", "docx"]
    # [1] 取第二个元素（索引为1），就是扩展名
    # lower() 转成小写，避免大小写问题
    extension = filename.rsplit('.', 1)[1].lower()
    
    # 检查扩展名是否在允许的集合中
    is_allowed = extension in ALLOWED_EXTENSIONS
    
    return has_dot and is_allowed


# =============================================================================
# 第四步：设置当前使用的知识库（集合）
# =============================================================================
# collection_name: 当前使用的知识库名称
# 在向量数据库中，每个上传的文档会创建一个"集合"（collection）
# 查询时会在当前集合中搜索相关内容
collection_name = 'demo'  # 默认集合名称为'demo'

# 检查uploads文件夹中是否已有上传的文档
# os.listdir() 列出文件夹中的所有文件
name_list = os.listdir(UPLOAD_FOLDER)

# 如果有文件，就把第一个文件名作为当前集合名称
# 这样系统启动后会自动使用最新上传的文档
if name_list:
    collection_name = name_list[0]


# =============================================================================
# 第五步：定义路由（处理用户请求）
# =============================================================================
# @app.route() 是装饰器，用来定义URL路由
# 当用户访问某个网址时，Flask会调用对应的函数

# -----------------------------------------------------------------------------
# 路由1: 文档上传页面
# -----------------------------------------------------------------------------
@app.route('/document_upload/', methods=['GET', 'POST'])
def document_upload():
    """
    【功能】处理文档上传
    
    【URL】/document_upload/
    
    【请求方式】
        GET: 用户打开上传页面（浏览器输入网址）
        POST: 用户提交表单（点击上传按钮）
    
    【流程】
        1. GET请求 -> 显示上传页面
        2. POST请求 -> 接收文件 -> 保存文件 -> 存入向量数据库
    """
    
    # ====== 情况1: 用户打开页面 (GET请求) ======
    if request.method == 'GET':
        # render_template() 渲染HTML模板
        # document_upload.html 是模板文件名，在templates文件夹中
        return render_template('document_upload.html')
    
    # ====== 情况2: 用户提交文件 (POST请求) ======
    elif request.method == 'POST':
        
        # --- 步骤1: 检查是否有文件被上传 ---
        # request.files 是一个字典，包含所有上传的文件
        # 'file' 是表单中文件输入框的name属性
        if 'file' not in request.files:
            flash('没有选择文件')  # 显示错误提示
            return redirect(request.url)  # 刷新页面
        
        # 获取上传的文件对象
        file = request.files['file']
        
        # --- 步骤2: 检查文件名是否为空 ---
        # 有时候浏览器会提交空文件（用户没选文件就点了上传）
        if file.filename == '':
            flash('没有选择文件')
            return redirect(request.url)
        
        # --- 步骤3: 检查文件类型并保存 ---
        # file 是文件对象
        # file.filename 是文件名
        if file and allowed_file(file.filename):
            
            # 获取文件名
            filename = file.filename
            print("接收到文件:", filename)  # 在控制台打印日志
            
            # 构建完整的保存路径
            # os.path.join() 智能地拼接路径（自动处理Windows和Linux的路径差异）
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("保存路径:", file_path)
            
            # 保存文件到uploads文件夹
            file.save(file_path)
            
            # --- 步骤4: 将文档存入向量数据库 ---
            # global 关键字表示我们要修改全局变量
            global collection_name
            
            # 使用正则表达式处理文件名，去除路径分隔符
            # re.split(r'[/\\]', filename) 按 / 或 \ 分割
            # [-1] 取最后一部分（真正的文件名）
            collection_name = re.split(r'[/\\]', filename)[-1]
            
            # 调用main.py中的save_to_db函数，将文档内容存入向量数据库
            # 参数1: 文件路径
            # 参数2: 集合名称（用文件名作为集合名）
            save_to_db(file_path, collection_name=collection_name)
            
            print(f"文档已存入向量数据库，集合名称: {collection_name}")
        
        # 上传完成后，刷新页面
        return redirect(request.url)
    
    # ====== 其他情况 ======
    else:
        return render_template('document_upload.html')


# -----------------------------------------------------------------------------
# 路由2: 聊天页面（首页）
# -----------------------------------------------------------------------------
@app.route('/')           # 根路径，访问 http://127.0.0.1:5000/ 会到这里
@app.route('/chat/', methods=['GET', 'POST'])  # 也可以访问 /chat/
def chat():
    """
    【功能】处理智能问答
    
    【URL】/ 或 /chat/
    
    【请求方式】
        GET: 用户打开聊天页面
        POST: 用户发送问题（AJAX请求）
    
    【流程】
        1. GET请求 -> 显示聊天页面
        2. POST请求 -> 接收问题 -> RAG检索 -> AI回答 -> 返回JSON
    """
    
    # ====== 情况1: 用户打开页面 (GET请求) ======
    if request.method == 'GET':
        return render_template('chat.html')
    
    # ====== 情况2: 用户发送问题 (POST请求) ======
    elif request.method == 'POST':
        # request.json 获取JSON格式的请求数据
        # .get('message') 获取message字段的值（用户输入的问题）
        message = request.json.get('message')
        print("用户提问:", message)
        
        # 检查问题是否为空
        if message:
            # 调用main.py中的rag_chat函数进行RAG问答
            # 参数1: 用户问题
            # 参数2: 集合名称（在哪个知识库中搜索）
            # 参数3: n_results=10 返回最相关的10条结果
            response, search_results = rag_chat(
                message, 
                collection_name=collection_name, 
                n_results=5
            )
            
            print('AI回答:', response)
            
            # 将检索结果合并成一个字符串（用于调试）
            final_search_results = '\n'.join(search_results)
            
            # 返回JSON格式的响应给前端
            # jsonify() 将Python字典转换为JSON字符串
            return jsonify({
                'response': response,           # AI的回答
                'search_results': search_results  # 检索到的相关文档片段
            })
        else:
            # 如果问题为空，返回错误信息
            # 400 是HTTP状态码，表示"错误的请求"
            return jsonify({'error': '请输入问题'}), 400
        
        # 这行代码实际上不会执行，因为上面已经有return了
        return redirect(request.url)


# -----------------------------------------------------------------------------
# 路由3: 切换当前使用的文档（集合）
# -----------------------------------------------------------------------------
@app.route('/collection/', methods=['GET', 'POST'])
def collection():
    """
    【功能】管理和切换知识库（文档集合）
    
    【URL】/collection/
    
    【请求方式】
        GET: 获取所有文档列表和当前使用的文档
        POST: 切换当前使用的文档
    
    【用途】
        当上传了多个文档时，用户可以切换使用哪个文档进行问答
    """
    
    # 声明使用全局变量，这样修改会影响到其他函数
    global collection_name
    
    # ====== 情况1: 获取文档列表 (GET请求) ======
    if request.method == 'GET':
        # 获取uploads文件夹中的所有文件名
        name_list = os.listdir(UPLOAD_FOLDER)
        
        # 如果有文件，返回文件列表和当前使用的文档名
        if name_list:
            return {
                'name_list': name_list,           # 所有文档名称列表
                'collection_name': collection_name  # 当前使用的文档名
            }
        
        # 如果没有文件，返回空列表
        return {
            'name_list': [],
            'collection_name': collection_name
        }
    
    # ====== 情况2: 切换文档 (POST请求) ======
    elif request.method == 'POST':
        # 获取前端传来的新文档名称
        new_collection = request.json.get('collection_name')
        
        # 更新全局变量
        collection_name = new_collection
        
        print('已切换到文档:', collection_name)
        
        # 重定向到聊天页面
        return redirect('/chat/')
    
    # 其他情况，重定向到聊天页面
    return redirect(request.url)


# =============================================================================
# 第六步：启动应用
# =============================================================================
# 当直接运行这个文件时（不是被导入时），执行以下代码
if __name__ == '__main__':
    # app.run() 启动Flask开发服务器
    # debug=True 开启调试模式：
    #   - 代码修改后自动重启
    #   - 出错时显示详细的错误信息
    #   - 不要在生产环境使用！
    # 默认port=5000
    app.run(debug=True, port=5556)
    
    # 启动后，在浏览器访问 http://127.0.0.1:5000
