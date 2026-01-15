# 多轮对话系统扩展方案

> **版本**: v1.0  
> **日期**: 2025-12-20  
> **状态**: 设计阶段

## 1. 概述

### 1.1 背景

当前 StockBench 框架采用**单轮决策**模式：
- 每次 LLM 调用是独立的 system_prompt + user_prompt
- Message 历史虽然存在，但未被实际使用
- 无法支持交互式决策、追问、解释等场景

### 1.2 目标

构建**生产级多轮对话系统**，支持：
- Agent 与用户的交互式对话
- Agent 间的协作对话
- 对话历史的持久化和检索
- 与 EpisodicMemory 的深度集成

### 1.3 设计原则

1. **渐进式升级**：保持现有单轮模式兼容
2. **状态管理**：清晰的对话状态机
3. **上下文压缩**：智能管理长对话的 token 消耗
4. **可追溯性**：完整的对话历史记录

---

## 2. 多轮对话场景分析

### 2.1 场景一：交互式决策

```
用户: 分析一下 AAPL 今天的走势
Agent: AAPL 今日上涨 2.3%，突破了 200 日均线...建议增持

用户: 为什么选择增持而不是观望？
Agent: 基于以下原因：1) 技术面突破确认 2) 成交量放大 3) 新闻面利好...

用户: 如果明天大盘下跌怎么办？
Agent: 建议设置止损位在 $175，如果跌破则减仓 50%...

用户: 好的，执行增持决策
Agent: 已记录增持决策，目标金额 $5000，置信度 0.85
```

### 2.2 场景二：Agent 间协作

```
Filter Agent → Decision Agent:
  "GS 需要基本面分析，原因：近期新闻涉及高管离职"

Decision Agent → Filter Agent:
  "收到，请提供 GS 的 PE 和市值数据"

Filter Agent → Decision Agent:
  "GS PE=15.8, 市值=2398亿美元，属于大盘价值股"

Decision Agent:
  "综合分析后，建议减持 GS..."
```

### 2.3 场景三：决策解释与回顾

```
用户: 为什么上周五卖出了 NVDA？
Agent: [检索 EpisodicMemory] 上周五卖出 NVDA 的原因是：
  1) 技术面出现顶背离
  2) 估值过高（PE=65）
  3) 市场情绪转向谨慎
  实际结果：卖出后 NVDA 下跌 8%，决策正确
```

---

## 3. 架构设计

### 3.1 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Conversation Manager                        │
│  管理对话生命周期、状态转换、上下文压缩                            │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Conversation   │ │  Message Store  │ │  Context        │
│  State Machine  │ │  (持久化存储)    │ │  Compressor     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PipelineContext                             │
│  conversation_history, memory, data_bus                         │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Filter Agent   │ │  Decision Agent │ │  User Interface │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 3.2 目录结构

```
stockbench/
├── conversation/
│   ├── __init__.py
│   ├── manager.py           # ConversationManager 主类
│   ├── state.py             # 对话状态机
│   ├── compressor.py        # 上下文压缩器
│   ├── store.py             # 对话存储
│   └── types.py             # 类型定义
├── core/
│   ├── message.py           # 现有：Message 类（增强）
│   └── pipeline_context.py  # 现有：增加对话管理集成
└── memory/
    └── layers/
        └── conversation.py  # 新增：对话记忆层
```

---

## 4. 核心组件设计

### 4.1 对话状态机

```python
# stockbench/conversation/state.py

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ConversationState(Enum):
    """对话状态枚举"""
    IDLE = auto()           # 空闲，等待输入
    ANALYZING = auto()      # 分析中
    AWAITING_CONFIRM = auto()  # 等待用户确认
    EXECUTING = auto()      # 执行决策中
    EXPLAINING = auto()     # 解释决策中
    COMPLETED = auto()      # 对话完成
    ERROR = auto()          # 错误状态


class ConversationIntent(Enum):
    """用户意图枚举"""
    ANALYZE = "analyze"           # 分析请求
    DECIDE = "decide"             # 决策请求
    EXPLAIN = "explain"           # 解释请求
    CONFIRM = "confirm"           # 确认执行
    CANCEL = "cancel"             # 取消
    FOLLOWUP = "followup"         # 追问
    REVIEW = "review"             # 回顾历史
    UNKNOWN = "unknown"           # 未知意图


@dataclass
class ConversationContext:
    """对话上下文"""
    id: str                                    # 对话ID
    state: ConversationState = ConversationState.IDLE
    intent: Optional[ConversationIntent] = None
    
    # 当前对话焦点
    focus_symbols: List[str] = field(default_factory=list)
    focus_date: Optional[str] = None
    
    # 待确认的决策
    pending_decisions: Dict[str, Dict] = field(default_factory=dict)
    
    # 对话元数据
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    turn_count: int = 0
    
    # 压缩后的历史摘要
    compressed_history: Optional[str] = None


class ConversationStateMachine:
    """
    对话状态机
    
    管理对话状态转换和意图识别
    """
    
    # 状态转换规则
    TRANSITIONS = {
        ConversationState.IDLE: {
            ConversationIntent.ANALYZE: ConversationState.ANALYZING,
            ConversationIntent.DECIDE: ConversationState.ANALYZING,
            ConversationIntent.REVIEW: ConversationState.EXPLAINING,
            ConversationIntent.EXPLAIN: ConversationState.EXPLAINING,
        },
        ConversationState.ANALYZING: {
            ConversationIntent.CONFIRM: ConversationState.EXECUTING,
            ConversationIntent.CANCEL: ConversationState.IDLE,
            ConversationIntent.FOLLOWUP: ConversationState.ANALYZING,
            ConversationIntent.EXPLAIN: ConversationState.EXPLAINING,
        },
        ConversationState.AWAITING_CONFIRM: {
            ConversationIntent.CONFIRM: ConversationState.EXECUTING,
            ConversationIntent.CANCEL: ConversationState.IDLE,
            ConversationIntent.FOLLOWUP: ConversationState.ANALYZING,
        },
        ConversationState.EXECUTING: {
            ConversationIntent.FOLLOWUP: ConversationState.ANALYZING,
            ConversationIntent.REVIEW: ConversationState.EXPLAINING,
        },
        ConversationState.EXPLAINING: {
            ConversationIntent.FOLLOWUP: ConversationState.EXPLAINING,
            ConversationIntent.DECIDE: ConversationState.ANALYZING,
            ConversationIntent.CONFIRM: ConversationState.EXECUTING,
        },
    }
    
    def __init__(self, context: ConversationContext):
        self.context = context
    
    def can_transition(self, intent: ConversationIntent) -> bool:
        """检查是否可以转换到目标状态"""
        current = self.context.state
        if current not in self.TRANSITIONS:
            return False
        return intent in self.TRANSITIONS[current]
    
    def transition(self, intent: ConversationIntent) -> bool:
        """执行状态转换"""
        if not self.can_transition(intent):
            return False
        
        new_state = self.TRANSITIONS[self.context.state][intent]
        self.context.state = new_state
        self.context.intent = intent
        self.context.last_activity = datetime.now()
        self.context.turn_count += 1
        
        return True
    
    def reset(self):
        """重置到初始状态"""
        self.context.state = ConversationState.IDLE
        self.context.intent = None
        self.context.pending_decisions.clear()
```

### 4.2 意图识别器

```python
# stockbench/conversation/intent.py

from typing import Tuple, List, Optional
from .state import ConversationIntent

class IntentClassifier:
    """
    意图分类器
    
    支持两种模式：
    1. 规则匹配（快速，无 API 调用）
    2. LLM 分类（准确，需要 API 调用）
    """
    
    # 规则匹配关键词
    INTENT_KEYWORDS = {
        ConversationIntent.ANALYZE: [
            "分析", "analyze", "看看", "怎么样", "走势", "趋势",
            "what about", "how is", "tell me about"
        ],
        ConversationIntent.DECIDE: [
            "决策", "decide", "买入", "卖出", "增持", "减持",
            "should i", "recommend", "建议"
        ],
        ConversationIntent.EXPLAIN: [
            "为什么", "why", "解释", "explain", "原因", "reason",
            "how come", "理由"
        ],
        ConversationIntent.CONFIRM: [
            "确认", "confirm", "执行", "execute", "好的", "ok",
            "yes", "是的", "同意", "agree"
        ],
        ConversationIntent.CANCEL: [
            "取消", "cancel", "不要", "算了", "no", "不",
            "never mind", "forget it"
        ],
        ConversationIntent.FOLLOWUP: [
            "那", "那么", "如果", "what if", "假设", "suppose",
            "还有", "另外", "and", "also"
        ],
        ConversationIntent.REVIEW: [
            "回顾", "review", "历史", "history", "之前", "上次",
            "过去", "previously", "last time"
        ],
    }
    
    def __init__(self, use_llm: bool = False, llm_client = None):
        self.use_llm = use_llm
        self.llm_client = llm_client
    
    def classify(self, text: str, context: 'ConversationContext' = None) -> Tuple[ConversationIntent, float]:
        """
        分类用户意图
        
        Args:
            text: 用户输入
            context: 对话上下文（用于上下文相关的意图识别）
            
        Returns:
            (意图, 置信度)
        """
        if self.use_llm and self.llm_client:
            return self._classify_with_llm(text, context)
        else:
            return self._classify_with_rules(text, context)
    
    def _classify_with_rules(self, text: str, context: 'ConversationContext') -> Tuple[ConversationIntent, float]:
        """规则匹配分类"""
        text_lower = text.lower()
        
        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return ConversationIntent.UNKNOWN, 0.0
        
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] / 3.0, 1.0)  # 归一化
        
        return best_intent, confidence
    
    def _classify_with_llm(self, text: str, context: 'ConversationContext') -> Tuple[ConversationIntent, float]:
        """LLM 分类"""
        prompt = f"""
        Classify the user's intent from the following message.
        
        Message: "{text}"
        
        Current conversation state: {context.state.name if context else 'IDLE'}
        Focus symbols: {context.focus_symbols if context else []}
        
        Choose one intent from:
        - ANALYZE: User wants to analyze a stock or market
        - DECIDE: User wants a trading decision/recommendation
        - EXPLAIN: User wants explanation for a decision
        - CONFIRM: User confirms to execute a decision
        - CANCEL: User cancels the current action
        - FOLLOWUP: User asks a follow-up question
        - REVIEW: User wants to review past decisions
        - UNKNOWN: Cannot determine intent
        
        Respond with JSON: {{"intent": "INTENT_NAME", "confidence": 0.0-1.0}}
        """
        
        # 调用 LLM（简化示例）
        # result = self.llm_client.generate_json(...)
        # return ConversationIntent[result["intent"]], result["confidence"]
        
        # 回退到规则匹配
        return self._classify_with_rules(text, context)
    
    def extract_symbols(self, text: str) -> List[str]:
        """从文本中提取股票代码"""
        import re
        # 匹配大写字母组成的 1-5 字符（股票代码格式）
        pattern = r'\b[A-Z]{1,5}\b'
        candidates = re.findall(pattern, text)
        
        # 过滤常见非股票代码的词
        stopwords = {'I', 'A', 'THE', 'AND', 'OR', 'IF', 'OK', 'YES', 'NO', 'PE', 'EPS'}
        return [c for c in candidates if c not in stopwords]
```

### 4.3 上下文压缩器

```python
# stockbench/conversation/compressor.py

from typing import List, Optional
from dataclasses import dataclass
from stockbench.core.message import Message

@dataclass
class CompressionResult:
    """压缩结果"""
    compressed_messages: List[Message]
    summary: str
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float


class ContextCompressor:
    """
    上下文压缩器
    
    当对话历史过长时，智能压缩以节省 token：
    1. 保留最近 N 轮完整对话
    2. 将更早的对话压缩为摘要
    3. 保留关键决策和确认信息
    
    压缩策略：
    - Sliding Window: 保留最近 N 条消息
    - Summarization: LLM 生成摘要
    - Selective: 保留关键消息，删除冗余
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        recent_turns_to_keep: int = 3,
        use_llm_summary: bool = True,
        llm_client = None
    ):
        self.max_tokens = max_tokens
        self.recent_turns_to_keep = recent_turns_to_keep
        self.use_llm_summary = use_llm_summary
        self.llm_client = llm_client
    
    def compress(self, messages: List[Message]) -> CompressionResult:
        """
        压缩对话历史
        
        Args:
            messages: 完整的消息列表
            
        Returns:
            CompressionResult: 压缩结果
        """
        from stockbench.core.message import estimate_tokens
        
        original_tokens = estimate_tokens(messages)
        
        # 如果未超过限制，不压缩
        if original_tokens <= self.max_tokens:
            return CompressionResult(
                compressed_messages=messages,
                summary="",
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                compression_ratio=1.0
            )
        
        # 分离系统消息、历史消息、最近消息
        system_msg = messages[0] if messages and messages[0].role == "system" else None
        non_system = messages[1:] if system_msg else messages
        
        # 保留最近 N 轮（每轮 = user + assistant）
        recent_count = self.recent_turns_to_keep * 2
        recent_messages = non_system[-recent_count:] if len(non_system) > recent_count else non_system
        older_messages = non_system[:-recent_count] if len(non_system) > recent_count else []
        
        # 压缩较早的消息
        if older_messages:
            if self.use_llm_summary and self.llm_client:
                summary = self._summarize_with_llm(older_messages)
            else:
                summary = self._summarize_with_rules(older_messages)
        else:
            summary = ""
        
        # 构建压缩后的消息列表
        compressed = []
        if system_msg:
            compressed.append(system_msg)
        
        if summary:
            # 将摘要作为系统消息的补充
            summary_msg = Message.system(f"[Previous conversation summary]\n{summary}")
            compressed.append(summary_msg)
        
        compressed.extend(recent_messages)
        
        compressed_tokens = estimate_tokens(compressed)
        
        return CompressionResult(
            compressed_messages=compressed,
            summary=summary,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        )
    
    def _summarize_with_rules(self, messages: List[Message]) -> str:
        """规则摘要（无 LLM）"""
        # 提取关键信息
        symbols_mentioned = set()
        decisions_made = []
        
        for msg in messages:
            # 提取股票代码
            import re
            symbols = re.findall(r'\b[A-Z]{1,5}\b', msg.content)
            symbols_mentioned.update(s for s in symbols if len(s) >= 2)
            
            # 提取决策关键词
            if any(kw in msg.content.lower() for kw in ['increase', 'decrease', 'buy', 'sell', 'hold']):
                decisions_made.append(msg.content[:100])
        
        summary_parts = []
        if symbols_mentioned:
            summary_parts.append(f"Discussed symbols: {', '.join(list(symbols_mentioned)[:10])}")
        if decisions_made:
            summary_parts.append(f"Decisions mentioned: {len(decisions_made)}")
        
        return "; ".join(summary_parts) if summary_parts else "Previous discussion context"
    
    def _summarize_with_llm(self, messages: List[Message]) -> str:
        """LLM 摘要"""
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content[:200]}" for msg in messages
        ])
        
        prompt = f"""
        Summarize the following conversation in 2-3 sentences.
        Focus on: symbols discussed, decisions made, key concerns raised.
        
        Conversation:
        {conversation_text}
        
        Summary:
        """
        
        # 调用 LLM
        # result = self.llm_client.generate(...)
        # return result
        
        # 回退到规则摘要
        return self._summarize_with_rules(messages)
```

### 4.4 对话管理器

```python
# stockbench/conversation/manager.py

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from stockbench.core.message import Message
from stockbench.core.pipeline_context import PipelineContext
from .state import ConversationContext, ConversationStateMachine, ConversationState, ConversationIntent
from .intent import IntentClassifier
from .compressor import ContextCompressor


@dataclass
class ConversationTurn:
    """单轮对话"""
    user_message: Message
    assistant_message: Optional[Message] = None
    intent: Optional[ConversationIntent] = None
    state_before: Optional[ConversationState] = None
    state_after: Optional[ConversationState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """
    对话管理器
    
    统一管理多轮对话的生命周期：
    - 意图识别
    - 状态转换
    - 上下文管理
    - 与 Agent 的交互
    - 对话持久化
    """
    
    def __init__(
        self,
        pipeline_ctx: PipelineContext,
        intent_classifier: Optional[IntentClassifier] = None,
        compressor: Optional[ContextCompressor] = None,
        auto_compress: bool = True,
        max_turns: int = 50
    ):
        self.pipeline_ctx = pipeline_ctx
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.compressor = compressor or ContextCompressor()
        self.auto_compress = auto_compress
        self.max_turns = max_turns
        
        # 初始化对话上下文
        self.context = ConversationContext(
            id=f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        self.state_machine = ConversationStateMachine(self.context)
        
        # 对话历史
        self.turns: List[ConversationTurn] = []
    
    def process_user_input(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        处理用户输入
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            (响应文本, 元数据)
        """
        # 1. 创建用户消息
        user_msg = Message.user(user_input)
        user_msg = user_msg.with_metadata(
            conversation_id=self.context.id,
            turn=self.context.turn_count
        )
        
        # 2. 意图识别
        intent, confidence = self.intent_classifier.classify(user_input, self.context)
        
        # 3. 提取股票代码
        symbols = self.intent_classifier.extract_symbols(user_input)
        if symbols:
            self.context.focus_symbols = symbols
        
        # 4. 状态转换
        state_before = self.context.state
        if self.state_machine.can_transition(intent):
            self.state_machine.transition(intent)
        
        # 5. 根据状态和意图生成响应
        response, metadata = self._generate_response(user_input, intent, confidence)
        
        # 6. 创建助手消息
        assistant_msg = Message.assistant(response)
        assistant_msg = assistant_msg.with_metadata(
            conversation_id=self.context.id,
            turn=self.context.turn_count,
            intent=intent.value,
            confidence=confidence
        )
        
        # 7. 记录对话轮次
        turn = ConversationTurn(
            user_message=user_msg,
            assistant_message=assistant_msg,
            intent=intent,
            state_before=state_before,
            state_after=self.context.state,
            metadata=metadata
        )
        self.turns.append(turn)
        
        # 8. 更新 PipelineContext 的对话历史
        self.pipeline_ctx.add_to_history(user_msg)
        self.pipeline_ctx.add_to_history(assistant_msg)
        
        # 9. 自动压缩
        if self.auto_compress and len(self.turns) > self.max_turns // 2:
            self._compress_history()
        
        return response, metadata
    
    def _generate_response(
        self, 
        user_input: str, 
        intent: ConversationIntent, 
        confidence: float
    ) -> Tuple[str, Dict[str, Any]]:
        """
        根据意图生成响应
        
        这里是核心逻辑，根据不同意图调用不同的处理器
        """
        metadata = {
            "intent": intent.value,
            "confidence": confidence,
            "state": self.context.state.name
        }
        
        if intent == ConversationIntent.ANALYZE:
            return self._handle_analyze(user_input, metadata)
        elif intent == ConversationIntent.DECIDE:
            return self._handle_decide(user_input, metadata)
        elif intent == ConversationIntent.EXPLAIN:
            return self._handle_explain(user_input, metadata)
        elif intent == ConversationIntent.CONFIRM:
            return self._handle_confirm(user_input, metadata)
        elif intent == ConversationIntent.CANCEL:
            return self._handle_cancel(user_input, metadata)
        elif intent == ConversationIntent.FOLLOWUP:
            return self._handle_followup(user_input, metadata)
        elif intent == ConversationIntent.REVIEW:
            return self._handle_review(user_input, metadata)
        else:
            return self._handle_unknown(user_input, metadata)
    
    def _handle_analyze(self, user_input: str, metadata: Dict) -> Tuple[str, Dict]:
        """处理分析请求"""
        symbols = self.context.focus_symbols
        if not symbols:
            return "请指定要分析的股票代码，例如：分析 AAPL", metadata
        
        # 调用 Agent 进行分析
        # 这里需要与现有的 Agent 系统集成
        # analysis = self.pipeline_ctx.get("analysis_result")
        
        response = f"正在分析 {', '.join(symbols)}...\n"
        response += "[此处将展示分析结果]"
        
        metadata["symbols"] = symbols
        return response, metadata
    
    def _handle_decide(self, user_input: str, metadata: Dict) -> Tuple[str, Dict]:
        """处理决策请求"""
        symbols = self.context.focus_symbols
        if not symbols:
            return "请指定要决策的股票代码", metadata
        
        # 调用 Decision Agent
        # decisions = decide_batch_dual_agent(...)
        
        # 将决策存入待确认
        # self.context.pending_decisions = decisions
        
        response = f"针对 {', '.join(symbols)} 的决策建议：\n"
        response += "[此处将展示决策建议]\n"
        response += "请确认是否执行？(输入'确认'或'取消')"
        
        # 转换到等待确认状态
        self.context.state = ConversationState.AWAITING_CONFIRM
        
        metadata["symbols"] = symbols
        return response, metadata
    
    def _handle_explain(self, user_input: str, metadata: Dict) -> Tuple[str, Dict]:
        """处理解释请求"""
        # 从 EpisodicMemory 检索相关决策
        if self.pipeline_ctx.memory_enabled:
            symbols = self.context.focus_symbols
            if symbols:
                history = self.pipeline_ctx.memory.episodes.get_history_for_prompt_dict(symbols, n=3)
                response = f"关于 {', '.join(symbols)} 的历史决策：\n"
                for symbol, decisions in history.items():
                    if decisions:
                        for d in decisions[:2]:
                            response += f"- {d['date']}: {d['action']}, 原因: {d.get('reasons', ['N/A'])[:2]}\n"
                    else:
                        response += f"- {symbol}: 无历史决策记录\n"
                return response, metadata
        
        return "请指定要解释的决策或股票代码", metadata
    
    def _handle_confirm(self, user_input: str, metadata: Dict) -> Tuple[str, Dict]:
        """处理确认请求"""
        if not self.context.pending_decisions:
            return "当前没有待确认的决策", metadata
        
        # 执行决策
        # execute_decisions(self.context.pending_decisions)
        
        response = "决策已确认执行！\n"
        response += f"执行了 {len(self.context.pending_decisions)} 个决策"
        
        # 清空待确认决策
        self.context.pending_decisions.clear()
        
        return response, metadata
    
    def _handle_cancel(self, user_input: str, metadata: Dict) -> Tuple[str, Dict]:
        """处理取消请求"""
        self.context.pending_decisions.clear()
        self.state_machine.reset()
        return "已取消当前操作", metadata
    
    def _handle_followup(self, user_input: str, metadata: Dict) -> Tuple[str, Dict]:
        """处理追问"""
        # 构建包含历史上下文的 prompt
        history_context = self._build_history_context()
        
        # 调用 LLM 回答追问
        # response = self.llm_client.generate(history_context + user_input)
        
        return f"[追问处理] 基于之前的对话，回答您的问题...", metadata
    
    def _handle_review(self, user_input: str, metadata: Dict) -> Tuple[str, Dict]:
        """处理回顾请求"""
        if self.pipeline_ctx.memory_enabled:
            # 从 EpisodicMemory 检索历史
            recent_episodes = self.pipeline_ctx.memory.episodes.query(days=7, limit=5)
            
            if recent_episodes:
                response = "最近 7 天的决策记录：\n"
                for ep in recent_episodes:
                    result_str = f" → {ep.actual_result:+.1f}%" if ep.actual_result else ""
                    response += f"- {ep.date} {ep.symbol}: {ep.action} ${ep.target_amount:.0f}{result_str}\n"
                return response, metadata
        
        return "无历史决策记录", metadata
    
    def _handle_unknown(self, user_input: str, metadata: Dict) -> Tuple[str, Dict]:
        """处理未知意图"""
        return "抱歉，我不太理解您的意思。您可以：\n- 分析股票：'分析 AAPL'\n- 获取决策：'AAPL 应该买入吗'\n- 解释决策：'为什么卖出 NVDA'", metadata
    
    def _build_history_context(self) -> str:
        """构建历史上下文"""
        if self.auto_compress:
            messages = [t.user_message for t in self.turns] + [t.assistant_message for t in self.turns if t.assistant_message]
            result = self.compressor.compress(messages)
            return result.summary
        else:
            return "\n".join([
                f"{t.user_message.role}: {t.user_message.content[:100]}"
                for t in self.turns[-5:]
            ])
    
    def _compress_history(self):
        """压缩历史"""
        messages = []
        for turn in self.turns:
            messages.append(turn.user_message)
            if turn.assistant_message:
                messages.append(turn.assistant_message)
        
        result = self.compressor.compress(messages)
        self.context.compressed_history = result.summary
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        return {
            "id": self.context.id,
            "state": self.context.state.name,
            "turn_count": self.context.turn_count,
            "focus_symbols": self.context.focus_symbols,
            "started_at": self.context.started_at.isoformat(),
            "last_activity": self.context.last_activity.isoformat(),
            "pending_decisions": len(self.context.pending_decisions),
        }
    
    def save_conversation(self) -> str:
        """保存对话到持久化存储"""
        # 实现对话持久化
        # 可以存储到 JSONL 文件或数据库
        pass
    
    def load_conversation(self, conversation_id: str) -> bool:
        """加载历史对话"""
        # 实现对话加载
        pass
```

---

## 5. 与现有系统集成

### 5.1 PipelineContext 增强

```python
# 在 stockbench/core/pipeline_context.py 中添加

@dataclass
class PipelineContext:
    # ... 现有字段 ...
    
    # 新增：对话管理器
    _conversation_manager: Optional['ConversationManager'] = field(default=None, repr=False)
    
    @property
    def conversation(self) -> 'ConversationManager':
        """获取对话管理器（延迟初始化）"""
        if self._conversation_manager is None:
            from stockbench.conversation import ConversationManager
            self._conversation_manager = ConversationManager(self)
        return self._conversation_manager
    
    def chat(self, user_input: str) -> str:
        """便捷的对话接口"""
        response, _ = self.conversation.process_user_input(user_input)
        return response
```

### 5.2 Agent 集成

```python
# 在 Agent 中使用对话上下文

def decide_batch_dual_agent(
    features_list: List[Dict],
    ctx: PipelineContext,
    # ... 其他参数
) -> Dict:
    # 检查是否在对话模式
    if ctx.conversation.context.state != ConversationState.IDLE:
        # 使用对话上下文增强 prompt
        conversation_context = ctx.conversation._build_history_context()
        # 将对话上下文添加到 system_prompt
        enhanced_system_prompt = f"{system_prompt}\n\n[Conversation Context]\n{conversation_context}"
    
    # ... 现有逻辑 ...
```

### 5.3 对话记忆层

```python
# stockbench/memory/layers/conversation.py

from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from stockbench.core.message import Message
from stockbench.memory.backends.file_backend import FileBackend


@dataclass
class ConversationRecord:
    """对话记录"""
    id: str
    started_at: datetime
    ended_at: Optional[datetime]
    turn_count: int
    symbols_discussed: List[str]
    decisions_made: List[str]
    summary: str
    messages: List[Dict]  # 序列化的 Message 列表


class ConversationMemory:
    """
    对话记忆层
    
    持久化存储对话历史，支持：
    - 按对话 ID 检索
    - 按股票代码检索相关对话
    - 按时间范围检索
    """
    
    def __init__(self, backend: FileBackend, data_dir: Path):
        self.backend = backend
        self.data_dir = Path(data_dir) / "conversations"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, record: ConversationRecord) -> str:
        """保存对话记录"""
        file_path = self.data_dir / f"conv_{record.started_at.strftime('%Y-%m')}.jsonl"
        self.backend.append_jsonl(file_path, self._to_dict(record))
        return record.id
    
    def get(self, conversation_id: str) -> Optional[ConversationRecord]:
        """按 ID 获取对话"""
        for file_path in self.data_dir.glob("conv_*.jsonl"):
            lines = self.backend.read_jsonl(file_path)
            for data in lines:
                if data.get("id") == conversation_id:
                    return self._from_dict(data)
        return None
    
    def search_by_symbol(self, symbol: str, limit: int = 10) -> List[ConversationRecord]:
        """按股票代码搜索相关对话"""
        results = []
        for file_path in sorted(self.data_dir.glob("conv_*.jsonl"), reverse=True):
            lines = self.backend.read_jsonl(file_path)
            for data in lines:
                if symbol in data.get("symbols_discussed", []):
                    results.append(self._from_dict(data))
                    if len(results) >= limit:
                        return results
        return results
    
    def _to_dict(self, record: ConversationRecord) -> Dict:
        return {
            "id": record.id,
            "started_at": record.started_at.isoformat(),
            "ended_at": record.ended_at.isoformat() if record.ended_at else None,
            "turn_count": record.turn_count,
            "symbols_discussed": record.symbols_discussed,
            "decisions_made": record.decisions_made,
            "summary": record.summary,
            "messages": record.messages,
        }
    
    def _from_dict(self, data: Dict) -> ConversationRecord:
        return ConversationRecord(
            id=data["id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            turn_count=data.get("turn_count", 0),
            symbols_discussed=data.get("symbols_discussed", []),
            decisions_made=data.get("decisions_made", []),
            summary=data.get("summary", ""),
            messages=data.get("messages", []),
        )
```

---

## 6. 配置设计

### 6.1 config.yaml 扩展

```yaml
conversation:
  enabled: true
  
  # 意图识别
  intent:
    use_llm: false  # 是否使用 LLM 进行意图识别
    confidence_threshold: 0.6
  
  # 上下文压缩
  compression:
    enabled: true
    max_tokens: 4000
    recent_turns_to_keep: 3
    use_llm_summary: false
  
  # 对话历史
  history:
    max_turns: 50
    persist: true
    storage_path: "storage/memory/conversations"
  
  # 状态机
  state_machine:
    timeout_minutes: 30  # 对话超时时间
    auto_reset_on_error: true
```

---

## 7. 使用示例

### 7.1 基本使用

```python
from stockbench.core.pipeline_context import PipelineContext
from stockbench.conversation import ConversationManager

# 创建 PipelineContext
ctx = PipelineContext(run_id="run_001", date="2025-03-12", ...)

# 使用便捷接口
response = ctx.chat("分析一下 AAPL 的走势")
print(response)

response = ctx.chat("为什么建议增持？")
print(response)

response = ctx.chat("好的，执行增持")
print(response)
```

### 7.2 高级使用

```python
from stockbench.conversation import ConversationManager, IntentClassifier, ContextCompressor

# 自定义配置
intent_classifier = IntentClassifier(use_llm=True, llm_client=llm_client)
compressor = ContextCompressor(max_tokens=8000, use_llm_summary=True)

manager = ConversationManager(
    pipeline_ctx=ctx,
    intent_classifier=intent_classifier,
    compressor=compressor
)

# 处理用户输入
response, metadata = manager.process_user_input("分析 AAPL 和 GOOGL")
print(f"Response: {response}")
print(f"Intent: {metadata['intent']}, Confidence: {metadata['confidence']}")

# 获取对话摘要
summary = manager.get_conversation_summary()
print(f"Conversation: {summary}")
```

---

## 8. 迁移计划

### 8.1 阶段一：基础设施（2-3 天）

- [ ] 创建 `stockbench/conversation/` 目录结构
- [ ] 实现 `ConversationState` 和 `ConversationStateMachine`
- [ ] 实现 `IntentClassifier`（规则版本）
- [ ] 编写单元测试

### 8.2 阶段二：核心功能（3-4 天）

- [ ] 实现 `ContextCompressor`
- [ ] 实现 `ConversationManager`
- [ ] 集成到 `PipelineContext`
- [ ] 实现 `ConversationMemory` 持久化

### 8.3 阶段三：Agent 集成（2-3 天）

- [ ] 修改 `decide_batch_dual_agent` 支持对话上下文
- [ ] 实现 `_handle_analyze`, `_handle_decide` 等处理器
- [ ] 与 EpisodicMemory 集成

### 8.4 阶段四：测试与优化（2-3 天）

- [ ] 端到端测试
- [ ] 性能优化
- [ ] 文档更新

---

## 9. 参考资料

- [LangChain Conversation Memory](https://python.langchain.com/docs/modules/memory/)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/chat)
- [Rasa Conversation Design](https://rasa.com/docs/rasa/conversation-driven-development/)
- [Microsoft Bot Framework State Management](https://docs.microsoft.com/en-us/azure/bot-service/bot-builder-concept-state)
