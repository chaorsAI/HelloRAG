# 其他指标优化：hybird_search、rebank
from time import sleep

from ConfigurableRAGEvaluator import  ConfigurableRAGEvaluator


# 评估数据
questions = ["如何使用安全带？", "车辆如何保养？", "座椅太热怎么办？"]
ground_truths = [
    '''调节座椅到合适位置，缓慢拉出安全带，将锁舌插入锁扣中，直到听见“咔哒”声。
    使腰部安全带应尽可能低的横跨于胯部。确保肩部安全带斜跨整个肩部，穿过胸部。
    将前排座椅安全带高度调整至合适的位置。
    请勿将座椅靠背太过向后倾斜。
    请在系紧安全带前检查锁扣插口是否存在异物（如：食物残渣等），若存在异物请及时取出。
    为确保安全带正常工作，请务必将安全带插入与之匹配的锁扣中。
    乘坐时，安全带必须拉紧，防止松垮，并确保其牢固贴身，无扭曲。
    切勿将安全带从您的后背绕过、从您的胳膊下面绕过或绕过您的颈部。安全带应远离您的面部和颈部，但不得从肩部滑落。
    如果安全带无法正常使用，请联系Lynk & Co领克中心进行处理。''',
    '''为了保持车辆处于最佳状态，建议您定期关注车辆状态，包括定期保养、洗车、内部清洁、外部清洁、轮胎的保养、低压蓄电池的保养等。''',
    '''有三种方式：1、通过中央显示屏，设置座椅加热强度或关闭座椅加热功能，
    在中央显示屏中点击座椅进入座椅加热控制界面，可在“关-低-中-高”之间循环。
    2、登录Lynk & Co App，按下前排座椅加热图标图标可以打开/关闭前排座椅加热。
    3、在中央显示屏中唤起空调控制界面然后点击舒适选项，降低座椅加热时间。'''
]

# 批量实验不同配置
configs = [
    {"chunk_size": 1024, "bm25_weight":0.5},                        # 混合检索
    {"chunk_size": 1024, "use_compress":True},                      # 压缩上下文
    {"chunk_size": 1024, "bm25_weight":0.5, "use_compress":True},   # 混合检索 + 压缩上下文
    {"chunk_size": 1024, "use_rerank":True},                        # 检索结果重排
]

all_results = []
for config in configs:
    sleep(20)
    print(f"------------ chunk_size = {config} ------------")
    evaluator = ConfigurableRAGEvaluator(
        pdf_path="../data/领克汽车使用手册.pdf",
        **config
    )
    result = evaluator.run(
        questions=questions,
        ground_truths=ground_truths,
    )
    result_str = evaluator.format_evaluation_result(show_details=False)
    print(result_str)
    all_results.append(result)