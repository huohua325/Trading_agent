#!/usr/bin/env python3
"""
Log Query Tool - 日志查询工具

快速查询和分析结构化 JSON 日志的命令行工具。

Usage:
    # 查找特定股票的决策
    python scripts/log_query.py --symbol AAPL --tag AGENT_DECISION
    
    # 查找失败的订单
    python scripts/log_query.py --status rejected --tag BT_ORDER
    
    # 查找高延迟的 LLM 调用
    python scripts/log_query.py --tag LLM_CALL --min-latency 3000
    
    # 导出 CSV
    python scripts/log_query.py --symbol AAPL --output decisions.csv
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import csv


class LogQuery:
    """日志查询器"""
    
    def __init__(self, log_dir: str = "logs/stockbench"):
        self.log_dir = Path(log_dir)
        
    def find_log_files(self, date: Optional[str] = None) -> List[Path]:
        """查找日志文件"""
        if not self.log_dir.exists():
            print(f"❌ Log directory not found: {self.log_dir}")
            return []
        
        if date:
            # 查找特定日期的日志
            pattern = f"{date}.log"
        else:
            # 查找所有日志文件
            pattern = "*.log"
        
        files = sorted(self.log_dir.glob(pattern))
        return files
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析日志行"""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return None
    
    def query(
        self,
        date: Optional[str] = None,
        symbol: Optional[str] = None,
        tag: Optional[str] = None,
        status: Optional[str] = None,
        agent_name: Optional[str] = None,
        action: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        min_latency: Optional[float] = None,
        max_latency: Optional[float] = None,
        cache_hit: Optional[bool] = None,
        level: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        查询日志
        
        Args:
            date: 日期 (YYYY-MM-DD)
            symbol: 股票代码
            tag: 日志标签 (e.g., AGENT_DECISION, BT_ORDER)
            status: 状态 (e.g., success, failed, rejected)
            agent_name: Agent 名称
            action: 决策动作 (hold, increase, decrease, close)
            min_confidence: 最小置信度
            max_confidence: 最大置信度
            min_latency: 最小延迟 (ms)
            max_latency: 最大延迟 (ms)
            cache_hit: 是否缓存命中
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
            limit: 结果数量限制
        
        Returns:
            匹配的日志条目列表
        """
        log_files = self.find_log_files(date)
        if not log_files:
            return []
        
        results = []
        
        for log_file in log_files:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = self.parse_log_line(line)
                    if not entry:
                        continue
                    
                    # 应用过滤条件
                    if symbol and entry.get('symbol') != symbol:
                        continue
                    
                    if tag and tag not in entry.get('message', ''):
                        continue
                    
                    if status and entry.get('status') != status:
                        continue
                    
                    if agent_name and entry.get('agent_name') != agent_name:
                        continue
                    
                    if action and entry.get('action') != action:
                        continue
                    
                    if level and entry.get('level') != level:
                        continue
                    
                    if min_confidence is not None:
                        conf = entry.get('confidence')
                        if conf is None or conf < min_confidence:
                            continue
                    
                    if max_confidence is not None:
                        conf = entry.get('confidence')
                        if conf is None or conf > max_confidence:
                            continue
                    
                    if min_latency is not None:
                        lat = entry.get('latency_ms') or entry.get('duration_ms')
                        if lat is None or lat < min_latency:
                            continue
                    
                    if max_latency is not None:
                        lat = entry.get('latency_ms') or entry.get('duration_ms')
                        if lat is None or lat > max_latency:
                            continue
                    
                    if cache_hit is not None and entry.get('cache_hit') != cache_hit:
                        continue
                    
                    results.append(entry)
                    
                    # 应用限制
                    if limit and len(results) >= limit:
                        return results
        
        return results
    
    def format_results(self, results: List[Dict[str, Any]], output_format: str = 'text') -> str:
        """格式化结果"""
        if not results:
            return "No matching logs found."
        
        if output_format == 'json':
            return json.dumps(results, indent=2, ensure_ascii=False)
        
        elif output_format == 'csv':
            # 收集所有字段
            all_keys = set()
            for entry in results:
                all_keys.update(entry.keys())
            
            output = []
            writer = csv.DictWriter(output, fieldnames=sorted(all_keys))
            
            # 写入 CSV
            import io
            csv_output = io.StringIO()
            csv_writer = csv.DictWriter(csv_output, fieldnames=sorted(all_keys))
            csv_writer.writeheader()
            csv_writer.writerows(results)
            return csv_output.getvalue()
        
        else:  # text format
            output_lines = []
            output_lines.append(f"Found {len(results)} matching log entries:\n")
            output_lines.append("=" * 80)
            
            for i, entry in enumerate(results, 1):
                output_lines.append(f"\n[{i}] {entry.get('time', 'N/A')} | {entry.get('level', 'INFO')} | {entry.get('message', '')}")
                
                # 显示关键字段
                if 'symbol' in entry:
                    output_lines.append(f"    Symbol: {entry['symbol']}")
                if 'action' in entry:
                    output_lines.append(f"    Action: {entry['action']}")
                if 'target_cash_amount' in entry:
                    output_lines.append(f"    Target: ${entry['target_cash_amount']:,.2f}")
                if 'confidence' in entry:
                    output_lines.append(f"    Confidence: {entry['confidence']:.2%}")
                if 'agent_name' in entry:
                    output_lines.append(f"    Agent: {entry['agent_name']}")
                if 'status' in entry:
                    output_lines.append(f"    Status: {entry['status']}")
                if 'duration_ms' in entry:
                    output_lines.append(f"    Duration: {entry['duration_ms']:.1f}ms")
                if 'latency_ms' in entry:
                    output_lines.append(f"    Latency: {entry['latency_ms']:.1f}ms")
                if 'error' in entry:
                    output_lines.append(f"    Error: {entry['error']}")
                
                output_lines.append("")
            
            output_lines.append("=" * 80)
            return "\n".join(output_lines)
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """保存结果到文件"""
        output_path = Path(output_file)
        
        # 根据文件扩展名确定格式
        if output_path.suffix == '.json':
            output_format = 'json'
        elif output_path.suffix == '.csv':
            output_format = 'csv'
        else:
            output_format = 'text'
        
        content = self.format_results(results, output_format)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Query structured JSON logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find all decisions for AAPL
  %(prog)s --symbol AAPL --tag AGENT_DECISION
  
  # Find failed orders
  %(prog)s --status rejected --tag BT_ORDER
  
  # Find high-latency LLM calls
  %(prog)s --tag LLM_CALL --min-latency 3000
  
  # Find low-confidence decisions
  %(prog)s --tag AGENT_DECISION --max-confidence 0.6
  
  # Export to CSV
  %(prog)s --symbol AAPL --output decisions.csv
  
  # Find cache misses
  %(prog)s --cache-hit false
        """
    )
    
    parser.add_argument('--log-dir', default='logs/stockbench', help='Log directory')
    parser.add_argument('--date', help='Date (YYYY-MM-DD)')
    parser.add_argument('--symbol', help='Stock symbol')
    parser.add_argument('--tag', help='Log tag (e.g., AGENT_DECISION, BT_ORDER)')
    parser.add_argument('--status', help='Status (e.g., success, failed, rejected)')
    parser.add_argument('--agent-name', help='Agent name')
    parser.add_argument('--action', help='Decision action (hold, increase, decrease, close)')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence')
    parser.add_argument('--max-confidence', type=float, help='Maximum confidence')
    parser.add_argument('--min-latency', type=float, help='Minimum latency (ms)')
    parser.add_argument('--max-latency', type=float, help='Maximum latency (ms)')
    parser.add_argument('--cache-hit', type=lambda x: x.lower() == 'true', help='Cache hit (true/false)')
    parser.add_argument('--level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    parser.add_argument('--limit', type=int, help='Limit number of results')
    parser.add_argument('--output', help='Output file (json, csv, or txt)')
    parser.add_argument('--format', choices=['text', 'json', 'csv'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    # 创建查询器
    querier = LogQuery(args.log_dir)
    
    # 执行查询
    results = querier.query(
        date=args.date,
        symbol=args.symbol,
        tag=args.tag,
        status=args.status,
        agent_name=args.agent_name,
        action=args.action,
        min_confidence=args.min_confidence,
        max_confidence=args.max_confidence,
        min_latency=args.min_latency,
        max_latency=args.max_latency,
        cache_hit=args.cache_hit,
        level=args.level,
        limit=args.limit
    )
    
    # 输出或保存结果
    if args.output:
        querier.save_results(results, args.output)
    else:
        print(querier.format_results(results, args.format))
    
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
