# 参考：https://docs.python.org/zh-cn/3/library/logging.html
# https://www.runoob.com/python3/python-logging.html
# https://blog.csdn.net/weixin_62528784/article/details/159978711?sharetype=blogdetail&sharerId=159978711&sharerefer=PC&sharesource=weixin_62528784&spm=1011.2480.3001.8118


# 1.
# 简单配置

# 全局配置，规定所有日志如何输出，是规则
logging.basicConfig(
    level=logging.INFO, # 日志级别≥INFO都会输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # 日志输出格式
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"/data2/logs/1_fetch_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log")
    ] # 输出到控制台+我们自己的指定文件
)
 
# 拿到当前脚本 / 模块专属的日志工具，是（真正用来打印日志的）工具
logger = logging.getLogger(__name__)
 
# 后面开始用这个模块logger输出
logger.info("开始下载PDB结构")
logger.warning("某个蛋白没有isoform")
logger.error("文件下载失败")


##################################################################################################

# 2.
# 稍微正规一点，封装成1个函数
# 主要是多次运行（但是有时间戳）
def _setup_logger() -> logging.Logger:
    """初始化脚本2日志器并同时输出到终端与日志文件。"""

    log_dir = Path("/data2/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / f"2_process_pdb_files_{stamp}.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


LOGGER = _setup_logger()
