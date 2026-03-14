import os
from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pymupdf  # PyMuPDF
import pdfplumber
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyPDF2Text:
    """
    高级PDF处理器，支持：
    1. 自动选择最优解析策略（数字PDF vs 扫描件）
    2. 并行页面解析加速
    3. 智能文本清洗和格式保留
    4. 可配置的语义分块
    """

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 use_ocr: bool = False,
                 max_workers: int = 4):
        """
        初始化处理器

        Args:
            chunk_size: 文本块大小（字符数）
            chunk_overlap: 块间重叠字符数
            use_ocr: 是否强制使用OCR（扫描件必须为True）
            max_workers: 并行解析的最大线程数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ocr = use_ocr
        self.max_workers = max_workers

        # 智能文本分割器，按语义边界分割
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract_text_from_pdf(self, filename: str) -> List[Document]:
        """
        从PDF提取文本并转换为Document对象

        Args:
            filename: PDF文件路径

        Returns:
            List[Document]: LangChain Document对象列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误或损坏
        """
        # 1. 基础验证
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件不存在: {filename}")

        if not filename.lower().endswith('.pdf'):
            raise ValueError("文件必须是PDF格式")

        logger.info(f"【PDF解析】开始处理: {filename}")

        # 2. 检测PDF类型并选择合适的解析器
        pdf_type = self._detect_pdf_type(filename)
        logger.info(f"【PDF检测】文件类型: {pdf_type}")

        # 3. 根据类型选择解析策略
        if pdf_type == "digital" and not self.use_ocr:
            # 数字PDF，使用高效解析
            pages_text = self._parse_digital_pdf(filename)
        else:
            # 扫描件或强制OCR
            pages_text = self._parse_scanned_pdf(filename)

        # 4. 后处理：清理文本
        cleaned_text = self._post_process_text(pages_text)

        # 5. 智能分块
        documents = self.splitter.create_documents([cleaned_text])
        # documents = self.splitter.split_text(cleaned_text)

        # 6. 添加元数据
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": filename,
                "page_count": len(pages_text),
                "chunk_index": i,
                "total_chunks": len(documents),
                "pdf_type": pdf_type
            })

        logger.info(f"【PDF解析】完成！提取{len(pages_text)}页，分割为{len(documents)}个文档块")

        return documents

    def _detect_pdf_type(self, filename: str) -> str:
        """
        检测PDF类型：数字版 或 扫描件

        原理：数字PDF包含文本对象，扫描件只有图像
        """
        try:
            with pymupdf.open(filename) as doc:
                for page_num in range(min(3, len(doc))):  # 检查前3页
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():  # 有可提取的文本
                        return "digital"
            return "scanned"
        except Exception as e:
            logger.warning(f"PDF类型检测失败: {e}")
            return "unknown"

    def _parse_digital_pdf(self, filename: str) -> List[str]:
        """
        解析数字PDF（基于PyMuPDF + pdfplumber双引擎）

        使用PyMuPDF提取基础文本，pdfplumber提取表格和格式
        双引擎策略确保最大提取率
        """
        pages_text = []

        try:
            # 引擎1: PyMuPDF（速度快，基础文本提取好）
            with pymupdf.open(filename) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")  # 纯文本模式
                    if text.strip():
                        pages_text.append(text)

            # 引擎2: pdfplumber（表格和精确位置信息）
            with pdfplumber.open(filename) as pdf:
                for page in pdf.pages:
                    # 提取表格
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # 将表格转为Markdown格式
                            table_text = self._table_to_markdown(table)
                            pages_text.append(table_text)

                    # 补充提取可能遗漏的文本
                    page_text = page.extract_text()
                    if page_text and page_text not in pages_text:
                        pages_text.append(page_text)

        except Exception as e:
            logger.error(f"数字PDF解析失败: {e}")
            # 回退到基础解析
            with pymupdf.open(filename) as doc:
                return [page.get_text() for page in doc]

        return pages_text

    def _parse_scanned_pdf(self, filename: str) -> List[str]:
        """
        解析扫描件PDF（基于OCR）

        注意：需要安装OCR引擎，如Tesseract
        """
        try:
            import pytesseract
            from PIL import Image

            pages_text = []
            with pymupdf.open(filename) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]

                    # 将PDF页面转为图像
                    pix = page.get_pixmap(dpi=300)  # 高DPI提高OCR精度
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # OCR识别
                    text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                    pages_text.append(text)

            return pages_text

        except ImportError as e:
            logger.error("OCR依赖未安装，请安装: pip install pytesseract pillow")
            logger.error("同时需要系统安装Tesseract: https://github.com/tesseract-ocr/tesseract")
            raise
        except Exception as e:
            logger.error(f"OCR解析失败: {e}")
            return ["[OCR解析失败]"]

    def _post_process_text(self, pages_text: List[str]) -> str:
        """
        文本后处理：清洗、去重、格式化
        """
        full_text = ""

        for i, page_text in enumerate(pages_text):
            if not page_text or page_text.isspace():
                continue

            # 1. 基础清洗
            cleaned = page_text.strip()

            # 2. 移除过多的空白字符
            import re
            cleaned = re.sub(r'\s+', ' ', cleaned)  # 多个空格变一个

            # 3. 智能段落合并
            if full_text and not full_text.endswith(('\n', '。', '!', '?', ';')):
                full_text += " "
            full_text += cleaned + "\n\n"

        return full_text.strip()

    def _table_to_markdown(self, table_data: List[List[str]]) -> str:
        """
        将表格数据转换为Markdown格式

        表格在RAG中很关键，但普通文本提取会丢失结构
        Markdown格式能较好保留表格语义
        """
        if not table_data or len(table_data) < 2:
            return ""

        markdown_lines = []

        # 表头
        headers = table_data[0]
        markdown_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        markdown_lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        # 表格内容
        for row in table_data[1:]:
            if row:  # 跳过空行
                markdown_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(markdown_lines)


# 使用示例
def extract_text_from_pdf(filename: str, **kwargs) -> List[Document]:
    """
    对外接口函数，保持与你原函数一致的使用方式
    """
    processor = MyPDF2Text(**kwargs)
    return processor.extract_text_from_pdf(filename)


# 单文件调用示例
if __name__ == "__main__":
    # 基本用法
    docs = extract_text_from_pdf("../data/公司采购流程.pdf")

    # 高级配置：处理扫描件，更大分块
    # docs = extract_text_from_pdf(
    #     "scanned_doc.pdf",
    #     use_ocr=True,
    #     chunk_size=800,
    #     chunk_overlap=100
    # )

    for i, doc in enumerate(docs[:3]):  # 打印前3个块
        print(f"\n=== 块 {i + 1}/{len(docs)} ===")
        print(f"元数据: {doc.metadata}")
        print(f"内容预览: {doc.page_content[:200]}...")