# -*- coding: utf-8 -*-

import pdfplumber
import re
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# 基于固定长度的切分
def fixed_size_chunking(pdf_path, chunk_size=500, chunk_overlap=50):
    """
    基于固定长度的 Naive RAG 切分代码
    """
    raw_text = ""

    # 1. 提取全文文本
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                raw_text += page_text + " "

    # 2. 使用递归字符分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

    # 3. 分割文本 - 使用 split_text 而不是 split_documents
    texts = text_splitter.split_text(raw_text)

    # 4. 转换为 Document 对象列表（Chroma.from_documents 需要）

    chunks = [Document(page_content=text) for text in texts]

    return chunks


# 基于段落的 RAG 切分（智能合并）
def process_medical_prd(pdf_path, min_chunk_size=500):
    """
    针对医药说明书的 RAG 智能段落分块策略
    策略：
    1. 对【】内字符串去空格处理
    2. 按段落分块（\n\n作为分隔符）
    3. 小于min_chunk_size的段落自动合并
    4. 表格不拆分，作为独立chunk
    """
    chunks = []
    current_section = "前言"
    current_content = []
    current_tables = []
    page_num = 0

    # 匹配说明书标准标题的正则表达式
    section_pattern = re.compile(r'^【\s*(.*?)\s*】')

    def convert_table_to_markdown(table):
        """将表格转换为 Markdown 格式"""
        if not table or len(table) < 1:
            return None
        
        df_temp = [["" if c is None else str(c).replace('\n', ' ').strip() for c in row] for row in table]
        df_temp = [row for row in df_temp if any(cell for cell in row)]
        
        if not df_temp:
            return None
        
        header = "| " + " | ".join(df_temp[0]) + " |"
        separator = "| " + " | ".join(["---"] * len(df_temp[0])) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in df_temp[1:]]
        
        return f"\n{header}\n{separator}\n" + "\n".join(rows) + "\n"

    def clean_section_title(title):
        """对【】内的字符串去空格处理"""
        return title.strip()

    def merge_small_paragraphs(paragraphs, min_size=min_chunk_size):
        """合并小于min_size的段落"""
        merged = []
        current_merged = ""
        
        for para in paragraphs:
            if len(current_merged) == 0:
                current_merged = para
            elif len(current_merged) + len(para) < min_size:
                current_merged += "\n\n" + para
            else:
                merged.append(current_merged)
                current_merged = para
        
        if current_merged:
            merged.append(current_merged)
        
        return merged

    def process_content():
        """处理当前内容，生成chunks"""
        nonlocal current_content, current_tables, current_section, page_num
        
        if not current_content and not current_tables:
            return []
        
        section_chunks = []
        
        # 处理表格：表格不拆分，作为独立chunk
        for table_md in current_tables:
            section_chunks.append({
                "type": "table",
                "header": current_section,
                "content": table_md,
                "page": page_num
            })
        
        # 处理文本：按段落分块
        if current_content:
            full_text = "\n".join(current_content)
            
            # 按段落分割（\n\n作为分隔符）
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            
            # 合并小于min_chunk_size的段落
            merged_paragraphs = merge_small_paragraphs(paragraphs, min_chunk_size)
            
            # 为每个段落chunk添加元数据
            for idx, para in enumerate(merged_paragraphs):
                section_chunks.append({
                    "type": "text",
                    "header": current_section,
                    "content": para,
                    "page": page_num,
                    "chunk_index": idx + 1,
                    "total_chunks": len(merged_paragraphs)
                })
        
        # 重置当前章节
        current_content = []
        current_tables = []
        
        return section_chunks

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # 1. 提取页面表格
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    table_md = convert_table_to_markdown(table)
                    if table_md:
                        current_tables.append(table_md)

            # 2. 提取文本
            text = page.extract_text()
            if not text:
                continue

            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 检查是否进入新章节
                match = section_pattern.match(line)
                if match:
                    # 保存旧章节
                    section_chunks = process_content()
                    chunks.extend(section_chunks)

                    # 清理章节标题，去空格
                    current_section = clean_section_title(match.group(1))
                    current_content = [line]
                else:
                    current_content.append(line)

        # 保存最后一个章节
        section_chunks = process_content()
        chunks.extend(section_chunks)

        # 将字典列表转换为 Document 对象列表
        documents = []
        for chunk in chunks:
            if chunk["type"] == "table":
                page_content = f"【{chunk['header']}】\n[数据表格]:\n{chunk['content']}"
                metadata = {
                    "header": chunk["header"],
                    "page": chunk["page"],
                    "source": "medical_prd",
                    "type": "table"
                }
            else:
                page_content = f"【{chunk['header']}】\n{chunk['content']}"
                metadata = {
                    "header": chunk["header"],
                    "page": chunk["page"],
                    "source": "medical_prd",
                    "type": "text"
                }
                
                if "chunk_index" in chunk:
                    metadata["chunk_index"] = chunk["chunk_index"]
                    metadata["total_chunks"] = chunk["total_chunks"]
            
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents


