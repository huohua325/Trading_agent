"""
日志标签标准化定义

统一的日志标签命名规范，用于 StockBench 项目的结构化日志。

标签分类:
- SYS_*: 系统层（初始化、配置、生命周期）
- DATA_*: 数据层（获取、缓存、验证）
- AGENT_*: Agent 层（过滤、决策、执行）
- BT_*: 回测层（引擎、订单、持仓、现金）
- LLM_*: LLM 层（调用、解析、缓存）
- MEM_*: Memory 层（保存、加载、回填）
- TOOL_*: 工具层（执行、失败）
"""

# ==================== 系统层标签 ====================
SYS_INIT = "SYS_INIT"              # 系统初始化
SYS_CONFIG = "SYS_CONFIG"          # 配置加载
SYS_START = "SYS_START"            # 系统启动
SYS_COMPLETE = "SYS_COMPLETE"      # 系统完成
SYS_ERROR = "SYS_ERROR"            # 系统错误

# ==================== 数据层标签 ====================
DATA_FETCH = "DATA_FETCH"          # 数据获取
DATA_CACHE = "DATA_CACHE"          # 缓存操作
DATA_LOAD = "DATA_LOAD"            # 数据加载
DATA_VALIDATE = "DATA_VALIDATE"    # 数据验证
DATA_ERROR = "DATA_ERROR"          # 数据错误

# ==================== Agent 层标签 ====================
AGENT_START = "AGENT_START"        # Agent 开始
AGENT_DONE = "AGENT_DONE"          # Agent 完成
AGENT_ERROR = "AGENT_ERROR"        # Agent 错误
AGENT_EXEC = "AGENT_EXEC"          # Agent 执行中
AGENT_FILTER = "AGENT_FILTER"      # 过滤 Agent
AGENT_DECISION = "AGENT_DECISION"  # 决策 Agent
AGENT_EXECUTOR = "AGENT_EXECUTOR"  # 执行器

# ==================== 回测层标签 ====================
BT_ENGINE = "BT_ENGINE"            # 回测引擎
BT_DAY = "BT_DAY"                  # 每日处理
BT_ORDER = "BT_ORDER"              # 订单执行
BT_ORDER_DETAIL = "BT_ORDER_DETAIL"  # 订单详细信息（DEBUG）
BT_CASH = "BT_CASH"                # 现金管理
BT_POSITION = "BT_POSITION"        # 持仓管理
BT_VALIDATE = "BT_VALIDATE"        # 持仓验证
BT_SHARES = "BT_SHARES"            # 份额计算
BT_PRICE = "BT_PRICE"              # 价格获取
BT_DIVIDEND = "BT_DIVIDEND"        # 股息处理
BT_SPLIT = "BT_SPLIT"              # 股票分割

# ==================== LLM 层标签 ====================
LLM_CALL = "LLM_CALL"              # LLM 调用
LLM_RESPONSE = "LLM_RESPONSE"      # LLM 响应
LLM_PARSE = "LLM_PARSE"            # LLM 解析
LLM_ERROR = "LLM_ERROR"            # LLM 错误
LLM_CACHE = "LLM_CACHE"            # LLM 缓存
LLM_STATS = "LLM_STATS"            # LLM 统计

# ==================== Memory 层标签 ====================
MEM_SAVE = "MEM_SAVE"              # Memory 保存
MEM_LOAD = "MEM_LOAD"              # Memory 加载
MEM_BACKFILL = "MEM_BACKFILL"      # Memory 回填
MEM_COMMIT = "MEM_COMMIT"          # Memory 提交
MEM_CLEAR = "MEM_CLEAR"            # Memory 清理
MEM_OP = "MEM_OP"                  # Memory 通用操作

# ==================== 工具层标签 ====================
TOOL_EXEC = "TOOL_EXEC"            # 工具执行
TOOL_FAIL = "TOOL_FAIL"            # 工具失败
TOOL_REGISTER = "TOOL_REGISTER"    # 工具注册

# ==================== 特征层标签 ====================
FEATURE_BUILD = "FEATURE_BUILD"    # 特征构建
FEATURE_NEWS = "FEATURE_NEWS"      # 新闻特征
FEATURE_FUND = "FEATURE_FUND"      # 基本面特征
FEATURE_TECH = "FEATURE_TECH"      # 技术面特征


# ==================== 标签映射表（旧 -> 新）====================
TAG_MIGRATION_MAP = {
    # Agent 层
    "[DUAL_AGENT]": AGENT_DECISION,
    "[FUNDAMENTAL_FILTER]": AGENT_FILTER,
    "[UNIFIED_EXECUTOR]": AGENT_EXECUTOR,
    
    # 回测层
    "[CASH_FLOW]": BT_CASH,
    "[CASH_UPDATE]": BT_CASH,
    "[CASH_PROTECTION]": BT_CASH,
    "[POSITION_VALIDATION]": BT_VALIDATE,
    "[SHARES_CALCULATION]": BT_SHARES,
    "[NEXT_DAY_PRICE]": BT_PRICE,
    "[DIVIDEND]": BT_DIVIDEND,
    
    # Memory 层
    "[PENDING_SAVE]": MEM_SAVE,
    "[MEMORY]": MEM_OP,
    
    # LLM 层
    "[LLM_CLIENT]": LLM_CALL,
    
    # 数据层
    "[DATA_FETCH]": DATA_FETCH,
    "[DATA_CACHE]": DATA_CACHE,
    
    # 系统层
    "[FILTER_STATS]": AGENT_FILTER,
    "[VALIDATION_ERROR]": BT_VALIDATE,
    "[VALIDATION_WARNING]": BT_VALIDATE,
    "[VALIDATION_OK]": BT_VALIDATE,
    "[HALLUCINATION_FILTER]": AGENT_DECISION,
    
    # 调试标签（冗余，直接移除）
    "[DEBUG]": "",  # 空字符串表示移除
}


def get_tag(old_tag: str) -> str:
    """
    获取标准化标签
    
    Args:
        old_tag: 旧标签（带或不带方括号）
        
    Returns:
        标准化标签
    """
    # 标准化输入：添加方括号
    if not old_tag.startswith("["):
        old_tag = f"[{old_tag}]"
    
    return TAG_MIGRATION_MAP.get(old_tag, old_tag)


def format_log_message(tag: str, message: str) -> str:
    """
    格式化日志消息
    
    Args:
        tag: 标签（不带方括号）
        message: 消息内容
        
    Returns:
        格式化后的消息：[TAG] message
    """
    return f"[{tag}] {message}"


# 导出所有标签常量
__all__ = [
    # 系统层
    "SYS_INIT", "SYS_CONFIG", "SYS_START", "SYS_COMPLETE", "SYS_ERROR",
    
    # 数据层
    "DATA_FETCH", "DATA_CACHE", "DATA_LOAD", "DATA_VALIDATE", "DATA_ERROR",
    
    # Agent 层
    "AGENT_START", "AGENT_DONE", "AGENT_ERROR", "AGENT_EXEC",
    "AGENT_FILTER", "AGENT_DECISION", "AGENT_EXECUTOR",
    
    # 回测层
    "BT_ENGINE", "BT_DAY", "BT_ORDER", "BT_ORDER_DETAIL",
    "BT_CASH", "BT_POSITION", "BT_VALIDATE", "BT_SHARES", "BT_PRICE",
    "BT_DIVIDEND", "BT_SPLIT",
    
    # LLM 层
    "LLM_CALL", "LLM_RESPONSE", "LLM_PARSE", "LLM_ERROR", "LLM_CACHE", "LLM_STATS",
    
    # Memory 层
    "MEM_SAVE", "MEM_LOAD", "MEM_BACKFILL", "MEM_COMMIT", "MEM_CLEAR", "MEM_OP",
    
    # 工具层
    "TOOL_EXEC", "TOOL_FAIL", "TOOL_REGISTER",
    
    # 特征层
    "FEATURE_BUILD", "FEATURE_NEWS", "FEATURE_FUND", "FEATURE_TECH",
    
    # 工具函数
    "TAG_MIGRATION_MAP", "get_tag", "format_log_message",
]
