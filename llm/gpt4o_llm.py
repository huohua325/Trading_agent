import json
import openai
from typing import Dict, List, Any, Optional
from .base_llm import BaseLLM
from ..actions.action_types import TradingAction, ActionType
import re


class GPT4oLLM(BaseLLM):
    """GPT-4o LLM实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.api_key = config.get("openai_api_key")
        self.api_base = config.get("openai_api_base", "https://api.openai.com/v1")
        self.model = config.get("openai_model", "gpt-4o")
        self.max_tokens = config.get("max_tokens", 1000)
        self.temperature = config.get("temperature", 0.1)
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # 设置OpenAI客户端
        openai.api_key = self.api_key
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    async def generate_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_status: Dict[str, Any],
        news_data: List[Dict[str, Any]],
        historical_context: Optional[Dict[str, Any]] = None,
        financial_data: Optional[Dict[str, Any]] = None,
        market_sentiment: Optional[Dict[str, Any]] = None
    ) -> TradingAction:
        """生成交易决策"""
        
        # 构建系统提示
        system_prompt = """你是一个专业的股票交易AI助手。你需要基于市场数据、投资组合状态、新闻信息、市场情绪和财务数据来做出交易决策。

        可用的交易行为:
        - buy: 买入股票
        - sell: 卖出股票  
        - hold: 观望/持有
        - get_info: 获取更多信息
        - get_news: 获取更多新闻

        你必须返回一个JSON格式的决策，包含以下字段:
        {
            "action_type": "buy/sell/hold/get_info/get_news",
            "symbol": "股票代码(如果适用)",
            "quantity": 数量(如果是买卖操作),
            "price": 目标价格(可选),
            "reason": "决策理由",
            "parameters": {额外参数}
        }

        请谨慎决策，考虑风险管理和投资组合平衡。不要在回复中包含任何额外的文本或代码块标记，只返回纯JSON对象。

        在做出决策时，请特别关注以下方面：
        1. 财务指标：如市盈率(PE)、每股收益(EPS)、股息收益率等
        2. 分析师推荐：买入/卖出/持有的比例和趋势
        3. 盈利惊喜：实际业绩与预期的对比
        4. 财务报表：资产负债表、利润表的关键指标
        5. 市场情绪：整体市场情绪、信心指数和风险水平
        6. 技术指标与市场数据的结合分析"""
        
        # 构建用户提示
        user_prompt = self._build_decision_prompt(
            market_data, portfolio_status, news_data, historical_context, financial_data, market_sentiment
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content.strip()
            print("\n===== LLM原始响应 =====")
            print(content)
            print("=======================\n")
            
            # 清理内容，移除可能的代码块标记和其他非JSON内容
            cleaned_content = self._clean_json_content(content)
            
            # 尝试解析JSON响应
            try:
                decision_data = json.loads(cleaned_content)
                print("JSON解析成功，决策数据:")
                print(json.dumps(decision_data, indent=2, ensure_ascii=False))
                
                # 确保action_type是小写的
                if "action_type" in decision_data and isinstance(decision_data["action_type"], str):
                    decision_data["action_type"] = decision_data["action_type"].lower()
                
            except json.JSONDecodeError as e:
                # 如果JSON解析失败，尝试提取关键信息
                print(f"JSON解析失败: {e}")
                decision_data = self._parse_fallback_response(content)
                print("使用备用解析，决策数据:")
                print(json.dumps(decision_data, indent=2, ensure_ascii=False))
            
            # 创建TradingAction对象
            action = TradingAction(**decision_data)
            print(f"创建的TradingAction: {action}")
            print(f"action_type: {action.action_type}, symbol: {action.symbol}, quantity: {action.quantity}")
            return action
            
        except Exception as e:
            # 如果出错，返回观望操作
            print(f"生成决策时出错: {str(e)}")
            return TradingAction(
                action_type="hold",
                reason=f"决策生成失败: {str(e)}"
            )
            
    async def analyze_market_sentiment(
        self,
        news_data: List[Dict[str, Any]],
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """分析市场情绪"""
        
        system_prompt = """你是一个专业的市场情绪分析师。基于提供的新闻数据分析市场情绪。

        请返回JSON格式的分析结果:
        {
            "overall_sentiment": "positive/negative/neutral",
            "confidence": 0.0-1.0,
            "key_factors": ["影响因素列表"],
            "risk_level": "low/medium/high",
            "recommendation": "建议"
        }"""
        
        news_prompt = self.format_news_prompt(news_data)
        if symbol:
            news_prompt += f"\n请特别关注与{symbol}相关的新闻。"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": news_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "overall_sentiment": "neutral",
                    "confidence": 0.5,
                    "key_factors": ["分析失败"],
                    "risk_level": "medium",
                    "recommendation": "谨慎观察"
                }
                
        except Exception as e:
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.0,
                "key_factors": [f"分析错误: {str(e)}"],
                "risk_level": "high",
                "recommendation": "暂停交易"
            }
    
    async def explain_decision(
        self,
        action: TradingAction,
        context: Dict[str, Any]
    ) -> str:
        """解释交易决策"""
        
        system_prompt = """你是一个交易分析师，需要清晰地解释交易决策的理由。
        用简洁明了的中文解释决策背后的逻辑和考虑因素。"""
        
        user_prompt = f"""请解释以下交易决策:
        
        决策: {action}
        理由: {action.reason}
        
        市场背景:
        {context.get('market_summary', '无市场信息')}
        
        请提供详细的决策解释。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"决策解释生成失败: {str(e)}"
    
    async def risk_assessment(
        self,
        portfolio_status: Dict[str, Any],
        proposed_action: TradingAction
    ) -> Dict[str, Any]:
        """风险评估"""
        
        system_prompt = """你是一个风险管理专家。评估拟议的交易行为的风险水平。

        返回JSON格式的风险评估:
        {
            "risk_level": "low/medium/high",
            "risk_score": 0.0-1.0,
            "risk_factors": ["风险因素列表"],
            "mitigation_suggestions": ["风险缓解建议"],
            "position_size_recommendation": "建议仓位大小",
            "stop_loss_recommendation": "止损建议"
        }"""
        
        portfolio_prompt = self.format_portfolio_prompt(portfolio_status)
        action_prompt = f"拟议操作: {proposed_action}"
        
        user_prompt = f"{portfolio_prompt}\n{action_prompt}\n\n请评估此操作的风险。"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "risk_level": "medium",
                    "risk_score": 0.5,
                    "risk_factors": ["评估失败"],
                    "mitigation_suggestions": ["暂停操作"],
                    "position_size_recommendation": "减少仓位",
                    "stop_loss_recommendation": "设置止损"
                }
                
        except Exception as e:
            return {
                "risk_level": "high",
                "risk_score": 1.0,
                "risk_factors": [f"风险评估错误: {str(e)}"],
                "mitigation_suggestions": ["停止交易"],
                "position_size_recommendation": "无操作",
                "stop_loss_recommendation": "立即平仓"
            }
    
    def _build_decision_prompt(
        self,
        market_data: Dict[str, Any],
        portfolio_status: Dict[str, Any],
        news_data: List[Dict[str, Any]],
        historical_context: Optional[Dict[str, Any]] = None,
        financial_data: Optional[Dict[str, Any]] = None,
        market_sentiment: Optional[Dict[str, Any]] = None
    ) -> str:
        """构建决策提示"""
        
        prompt = "请基于以下信息做出交易决策:\n\n"
        
        # 市场数据
        prompt += self.format_market_data_prompt(market_data)
        prompt += "\n"
        
        # 投资组合状态
        prompt += self.format_portfolio_prompt(portfolio_status)
        prompt += "\n"
        
        # 新闻数据
        prompt += self.format_news_prompt(news_data)
        prompt += "\n"
        
        # 财务数据
        if financial_data:
            prompt += self.format_financial_data_prompt(financial_data)
            prompt += "\n"
        
        # 市场情绪
        if market_sentiment:
            prompt += self.format_market_sentiment_prompt(market_sentiment)
        prompt += "\n"
        
        # 历史上下文
        if historical_context:
            prompt += "历史上下文:\n"
            prompt += f"- 最近表现: {historical_context.get('recent_performance', 'N/A')}\n"
            prompt += f"- 交易历史: {historical_context.get('trade_history', 'N/A')}\n\n"
        
        prompt += "请提供你的交易决策（JSON格式）:"
        
        return prompt
    
    def format_financial_data_prompt(self, financial_data: Dict[str, Any]) -> str:
        """格式化财务数据为提示文本"""
        prompt = "财务数据:\n"
        
        # 基本财务指标
        if "key_metrics" in financial_data:
            metrics = financial_data["key_metrics"]
            prompt += "基本财务指标:\n"
            for name, value in metrics.items():
                if value is not None:
                    prompt += f"- {name}: {value}\n"
        
        # 盈利惊喜
        if "earnings_surprises" in financial_data and financial_data["earnings_surprises"]:
            earnings = financial_data["earnings_surprises"]
            if isinstance(earnings, list) and len(earnings) > 0:
                prompt += "\n盈利惊喜数据:\n"
                for i, quarter in enumerate(earnings[:2]):  # 只显示最近两个季度
                    prompt += f"- {quarter.get('period', 'N/A')}: 预期 ${quarter.get('estimate', 'N/A')}, 实际 ${quarter.get('actual', 'N/A')}, 惊喜 {quarter.get('surprisePercent', 'N/A')}%\n"
        
        # 分析师推荐
        if "recommendation_trends" in financial_data and financial_data["recommendation_trends"]:
            trends = financial_data["recommendation_trends"]
            if isinstance(trends, list) and len(trends) > 0:
                latest = trends[0]
                prompt += "\n分析师推荐:\n"
                prompt += f"- 强烈买入: {latest.get('strongBuy', 'N/A')}\n"
                prompt += f"- 买入: {latest.get('buy', 'N/A')}\n"
                prompt += f"- 持有: {latest.get('hold', 'N/A')}\n"
                prompt += f"- 卖出: {latest.get('sell', 'N/A')}\n"
                prompt += f"- 强烈卖出: {latest.get('strongSell', 'N/A')}\n"
        
        # 财务报表摘要
        if "reported_financials" in financial_data and "data" in financial_data["reported_financials"]:
            data = financial_data["reported_financials"]["data"]
            if isinstance(data, list) and len(data) > 0 and "report" in data[0]:
                prompt += "\n财务报表摘要:\n"
                
                # 资产负债表摘要
                if "bs" in data[0]["report"]:
                    bs_items = data[0]["report"]["bs"]
                    key_bs_items = [
                        item for item in bs_items 
                        if any(keyword in item.get("label", "").lower() 
                              for keyword in ["assets", "liabilities", "equity", "cash", "debt", "revenue", "income", "profit"])
                    ]
                    if key_bs_items:
                        prompt += "资产负债表关键项目:\n"
                        for item in key_bs_items[:5]:  # 只显示前5个关键项目
                            prompt += f"- {item.get('label')}: {item.get('value')}\n"
                
                # 利润表摘要
                if "ic" in data[0]["report"]:
                    ic_items = data[0]["report"]["ic"]
                    key_ic_items = [
                        item for item in ic_items 
                        if any(keyword in item.get("label", "").lower() 
                              for keyword in ["revenue", "income", "profit", "margin", "earnings", "sales"])
                    ]
                    if key_ic_items:
                        prompt += "利润表关键项目:\n"
                        for item in key_ic_items[:5]:  # 只显示前5个关键项目
                            prompt += f"- {item.get('label')}: {item.get('value')}\n"
        
        return prompt
    
    def format_market_sentiment_prompt(self, market_sentiment: Dict[str, Any]) -> str:
        """格式化市场情绪数据为提示文本"""
        prompt = "市场情绪分析:\n"
        
        # 整体情绪
        overall_sentiment = market_sentiment.get("overall_sentiment", "neutral")
        sentiment_map = {
            "positive": "积极",
            "negative": "消极",
            "neutral": "中性",
            "bullish": "看涨",
            "bearish": "看跌"
        }
        translated_sentiment = sentiment_map.get(overall_sentiment, overall_sentiment)
        prompt += f"- 整体情绪: {translated_sentiment}\n"
        
        # 信心指数
        confidence = market_sentiment.get("confidence", 0.5)
        prompt += f"- 信心指数: {confidence:.2f}\n"
        
        # 风险水平
        risk_level = market_sentiment.get("risk_level", "medium")
        risk_map = {
            "low": "低",
            "medium": "中",
            "high": "高"
        }
        translated_risk = risk_map.get(risk_level, risk_level)
        prompt += f"- 风险水平: {translated_risk}\n"
        
        # 关键因素
        key_factors = market_sentiment.get("key_factors", [])
        if key_factors:
            prompt += "- 关键因素:\n"
            for factor in key_factors[:3]:  # 只显示前3个因素
                prompt += f"  * {factor}\n"
        
        # 建议
        recommendation = market_sentiment.get("recommendation", "")
        if recommendation:
            prompt += f"- 建议: {recommendation}\n"
        
        return prompt
    
    def _parse_fallback_response(self, content: str) -> Dict[str, Any]:
        """解析非JSON格式的响应"""
        
        # 简单的关键词匹配来确定操作类型
        content_lower = content.lower()
        
        if "buy" in content_lower or "买入" in content_lower:
            action_type = "buy"
        elif "sell" in content_lower or "卖出" in content_lower:
            action_type = "sell"
        elif "hold" in content_lower or "持有" in content_lower or "观望" in content_lower:
            action_type = "hold"
        elif "info" in content_lower or "信息" in content_lower:
            action_type = "get_info"
        elif "news" in content_lower or "新闻" in content_lower:
            action_type = "get_news"
        else:
            action_type = "hold"
        
        # 尝试提取股票代码和数量
        symbol = None
        quantity = None
        
        # 尝试提取股票代码
        symbol_match = re.search(r'(?:symbol|股票|代码)[：:]\s*["\']?([A-Z]{1,5})["\']?', content)
        if symbol_match:
            symbol = symbol_match.group(1)
        
        # 尝试提取数量
        quantity_match = re.search(r'(?:quantity|数量|股数)[：:]\s*(\d+)', content)
        if quantity_match:
            try:
                quantity = int(quantity_match.group(1))
            except ValueError:
                pass
        
        return {
            "action_type": action_type,
            "symbol": symbol,
            "quantity": quantity,
            "reason": content[:200],  # 取前200个字符作为理由
        } 

    def _clean_json_content(self, content: str) -> str:
        """清理LLM返回的内容，提取有效的JSON"""
        # 移除可能的代码块标记
        content = re.sub(r'```(?:json)?|```', '', content)
        
        # 处理可能的重复字段问题
        # 这里使用一个简单的方法：找到第一个完整的JSON对象
        try:
            # 尝试找到第一个{和最后一个}之间的内容
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                content = content[start:end+1]
                
            # 使用正则表达式处理重复的字段
            # 例如："reason": "part1", "reason": "part2", "reason": "part3"
            # 转换为："reason": "part1 part2 part3"
            pattern = r'"(\w+)"\s*:\s*"([^"]*)"\s*,\s*"(\1)"\s*:\s*"([^"]*)"\s*,\s*"(\1)"\s*:\s*"([^"]*)"'
            replacement = r'"\1": "\2 \4 \6"'
            content = re.sub(pattern, replacement, content)
            
            # 处理两次重复的情况
            pattern = r'"(\w+)"\s*:\s*"([^"]*)"\s*,\s*"(\1)"\s*:\s*"([^"]*)"'
            replacement = r'"\1": "\2 \4"'
            content = re.sub(pattern, replacement, content)
            
            # 将action_type字段的值转换为小写
            # 例如："action_type": "BUY" 转换为 "action_type": "buy"
            pattern = r'"action_type"\s*:\s*"([^"]*)"'
            def lowercase_action(match):
                return f'"action_type": "{match.group(1).lower()}"'
            content = re.sub(pattern, lowercase_action, content)
            
            return content
        except Exception as e:
            print(f"清理JSON内容时出错: {e}")
            return content 