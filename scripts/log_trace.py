#!/usr/bin/env python3
"""
Log Trace Visualizer - 执行链路追踪可视化工具

可视化展示 Agent 执行链路和依赖关系。

Usage:
    # 追踪特定 run_id 的执行
    python scripts/log_trace.py --run-id backtest_20251215_001
    
    # 追踪特定日期的执行
    python scripts/log_trace.py --date 2025-12-15
    
    # 生成 HTML 可视化
    python scripts/log_trace.py --run-id backtest_20251215_001 --html trace.html
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict


class ExecutionTrace:
    """执行追踪记录"""
    
    def __init__(self, run_id: str, date: str):
        self.run_id = run_id
        self.date = date
        self.agents = []
        self.decisions = []
        self.orders = []
        self.llm_calls = []
        self.data_fetches = []
        self.errors = []
    
    def add_agent(self, entry: Dict[str, Any]):
        """添加 Agent 记录"""
        self.agents.append({
            'time': entry.get('time'),
            'agent_name': entry.get('agent_name'),
            'status': entry.get('status'),
            'duration_ms': entry.get('duration_ms'),
            'input_count': entry.get('input_count'),
            'output_count': entry.get('output_count'),
            'error': entry.get('error'),
        })
    
    def add_decision(self, entry: Dict[str, Any]):
        """添加决策记录"""
        self.decisions.append({
            'time': entry.get('time'),
            'symbol': entry.get('symbol'),
            'action': entry.get('action'),
            'confidence': entry.get('confidence'),
            'target_cash_amount': entry.get('target_cash_amount'),
        })
    
    def add_order(self, entry: Dict[str, Any]):
        """添加订单记录"""
        self.orders.append({
            'time': entry.get('time'),
            'symbol': entry.get('symbol'),
            'side': entry.get('side'),
            'qty': entry.get('qty'),
            'status': entry.get('status'),
        })
    
    def add_llm_call(self, entry: Dict[str, Any]):
        """添加 LLM 调用记录"""
        self.llm_calls.append({
            'time': entry.get('time'),
            'model': entry.get('model'),
            'latency_ms': entry.get('latency_ms'),
            'total_tokens': entry.get('total_tokens'),
            'cache_hit': entry.get('cache_hit'),
        })
    
    def add_data_fetch(self, entry: Dict[str, Any]):
        """添加数据获取记录"""
        self.data_fetches.append({
            'time': entry.get('time'),
            'data_type': entry.get('data_type'),
            'source': entry.get('source'),
            'cache_hit': entry.get('cache_hit'),
        })
    
    def add_error(self, entry: Dict[str, Any]):
        """添加错误记录"""
        self.errors.append({
            'time': entry.get('time'),
            'level': entry.get('level'),
            'message': entry.get('message'),
            'error': entry.get('error'),
        })


class TraceVisualizer:
    """链路追踪可视化器"""
    
    def __init__(self, log_dir: str = "logs/stockbench"):
        self.log_dir = Path(log_dir)
    
    def find_log_files(self, date: Optional[str] = None) -> List[Path]:
        """查找日志文件"""
        if not self.log_dir.exists():
            print(f"❌ Log directory not found: {self.log_dir}")
            return []
        
        if date:
            pattern = f"{date}.log"
        else:
            pattern = "*.log"
        
        files = sorted(self.log_dir.glob(pattern))
        return files
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析日志行"""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return None
    
    def collect_trace(self, run_id: Optional[str] = None, date: Optional[str] = None) -> List[ExecutionTrace]:
        """收集执行追踪"""
        log_files = self.find_log_files(date)
        if not log_files:
            return []
        
        traces = {}  # run_id -> ExecutionTrace
        
        for log_file in log_files:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = self.parse_log_line(line)
                    if not entry:
                        continue
                    
                    entry_run_id = entry.get('run_id')
                    entry_date = entry.get('date')
                    
                    # 如果指定了 run_id，只收集该 run_id 的记录
                    if run_id and entry_run_id != run_id:
                        continue
                    
                    # 创建或获取 trace
                    if entry_run_id and entry_run_id not in traces:
                        traces[entry_run_id] = ExecutionTrace(entry_run_id, entry_date or date or 'unknown')
                    
                    trace = traces.get(entry_run_id or 'unknown')
                    if not trace:
                        continue
                    
                    # 分类记录
                    if entry.get('agent_name'):
                        trace.add_agent(entry)
                    
                    if 'AGENT_DECISION' in entry.get('message', ''):
                        trace.add_decision(entry)
                    
                    if 'BT_ORDER' in entry.get('message', ''):
                        trace.add_order(entry)
                    
                    if 'LLM_CALL' in entry.get('message', '') or 'LLM_CACHE' in entry.get('message', ''):
                        trace.add_llm_call(entry)
                    
                    if 'DATA_FETCH' in entry.get('message', '') or 'DATA_CACHE' in entry.get('message', ''):
                        trace.add_data_fetch(entry)
                    
                    if entry.get('level') in ['ERROR', 'WARNING'] and entry.get('error'):
                        trace.add_error(entry)
        
        return list(traces.values())
    
    def format_text_trace(self, trace: ExecutionTrace) -> str:
        """格式化文本追踪"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"🔍 EXECUTION TRACE: {trace.run_id}")
        lines.append(f"📅 Date: {trace.date}")
        lines.append("=" * 80)
        lines.append("")
        
        # Agent 执行时间线
        if trace.agents:
            lines.append("🤖 AGENT EXECUTION TIMELINE")
            lines.append("-" * 80)
            
            for agent in sorted(trace.agents, key=lambda x: x['time'] or ''):
                status_icon = "✅" if agent['status'] == 'success' else "❌" if agent['status'] == 'failed' else "⏳"
                lines.append(f"{agent['time']} | {status_icon} {agent['agent_name']}")
                
                if agent.get('duration_ms'):
                    lines.append(f"  Duration: {agent['duration_ms']:.1f}ms")
                if agent.get('input_count'):
                    lines.append(f"  Input: {agent['input_count']} items")
                if agent.get('output_count'):
                    lines.append(f"  Output: {agent['output_count']} items")
                if agent.get('error'):
                    lines.append(f"  ❌ Error: {agent['error'][:100]}")
                lines.append("")
            
            lines.append("")
        
        # 决策汇总
        if trace.decisions:
            lines.append("📈 DECISIONS SUMMARY")
            lines.append("-" * 80)
            
            action_counts = defaultdict(int)
            for decision in trace.decisions:
                action_counts[decision['action']] += 1
            
            lines.append(f"Total Decisions: {len(trace.decisions)}")
            for action, count in sorted(action_counts.items()):
                lines.append(f"  - {action}: {count}")
            
            # 列出高置信度决策
            high_conf = [d for d in trace.decisions if d.get('confidence', 0) > 0.8]
            if high_conf:
                lines.append(f"\nHigh Confidence Decisions ({len(high_conf)}):")
                for d in high_conf[:10]:  # 只显示前 10 个
                    lines.append(f"  - {d['symbol']}: {d['action']} (confidence={d['confidence']:.1%})")
            
            lines.append("")
        
        # LLM 调用汇总
        if trace.llm_calls:
            lines.append("🧠 LLM CALLS SUMMARY")
            lines.append("-" * 80)
            
            total_calls = len(trace.llm_calls)
            cache_hits = sum(1 for call in trace.llm_calls if call.get('cache_hit'))
            total_tokens = sum(call.get('total_tokens', 0) for call in trace.llm_calls)
            total_latency = sum(call.get('latency_ms', 0) for call in trace.llm_calls)
            
            lines.append(f"Total Calls: {total_calls}")
            lines.append(f"Cache Hits: {cache_hits} ({cache_hits/total_calls:.1%})")
            lines.append(f"Total Tokens: {total_tokens:,}")
            lines.append(f"Total Latency: {total_latency:.1f}ms")
            lines.append(f"Avg Latency: {total_latency/total_calls:.1f}ms")
            lines.append("")
        
        # 数据获取汇总
        if trace.data_fetches:
            lines.append("📦 DATA FETCHES SUMMARY")
            lines.append("-" * 80)
            
            total_fetches = len(trace.data_fetches)
            cache_hits = sum(1 for fetch in trace.data_fetches if fetch.get('cache_hit'))
            
            lines.append(f"Total Fetches: {total_fetches}")
            lines.append(f"Cache Hits: {cache_hits} ({cache_hits/total_fetches:.1%})")
            lines.append("")
        
        # 错误汇总
        if trace.errors:
            lines.append("⚠️  ERRORS & WARNINGS")
            lines.append("-" * 80)
            
            for error in trace.errors[:20]:  # 最多显示 20 个
                lines.append(f"{error['time']} | {error['level']}")
                lines.append(f"  Message: {error['message']}")
                if error.get('error'):
                    lines.append(f"  Error: {error['error'][:100]}")
                lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_html_trace(self, trace: ExecutionTrace) -> str:
        """生成 HTML 可视化追踪"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Execution Trace: {trace.run_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .timeline {{
            position: relative;
            padding-left: 40px;
        }}
        .timeline-item {{
            position: relative;
            padding: 15px;
            margin-bottom: 20px;
            background: #f9f9f9;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
        }}
        .timeline-item.failed {{
            border-left-color: #f44336;
        }}
        .timeline-item .time {{
            color: #888;
            font-size: 0.9em;
        }}
        .timeline-item .agent-name {{
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }}
        .stat-card .label {{
            color: #888;
            font-size: 0.9em;
        }}
        .stat-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .decision-list {{
            list-style: none;
            padding: 0;
        }}
        .decision-list li {{
            padding: 10px;
            margin: 5px 0;
            background: #f9f9f9;
            border-radius: 4px;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .badge.increase {{ background: #4CAF50; color: white; }}
        .badge.decrease {{ background: #ff9800; color: white; }}
        .badge.hold {{ background: #9E9E9E; color: white; }}
        .badge.close {{ background: #f44336; color: white; }}
        .error-list {{
            background: #fff3e0;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #ff9800;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Execution Trace: {trace.run_id}</h1>
        <p>📅 Date: {trace.date}</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="label">Agents Executed</div>
                <div class="value">{len(trace.agents)}</div>
            </div>
            <div class="stat-card">
                <div class="label">Decisions Made</div>
                <div class="value">{len(trace.decisions)}</div>
            </div>
            <div class="stat-card">
                <div class="label">LLM Calls</div>
                <div class="value">{len(trace.llm_calls)}</div>
            </div>
            <div class="stat-card">
                <div class="label">Errors/Warnings</div>
                <div class="value">{len(trace.errors)}</div>
            </div>
        </div>
        
        <h2>🤖 Agent Execution Timeline</h2>
        <div class="timeline">
"""
        
        # Agent 时间线
        for agent in sorted(trace.agents, key=lambda x: x['time'] or ''):
            failed_class = ' failed' if agent['status'] == 'failed' else ''
            status_icon = "✅" if agent['status'] == 'success' else "❌" if agent['status'] == 'failed' else "⏳"
            
            html += f"""
            <div class="timeline-item{failed_class}">
                <div class="time">{agent['time']}</div>
                <div class="agent-name">{status_icon} {agent['agent_name']}</div>
"""
            if agent.get('duration_ms'):
                html += f"                <div>Duration: {agent['duration_ms']:.1f}ms</div>\n"
            if agent.get('input_count'):
                html += f"                <div>Input: {agent['input_count']} items</div>\n"
            if agent.get('output_count'):
                html += f"                <div>Output: {agent['output_count']} items</div>\n"
            if agent.get('error'):
                html += f"                <div style='color: #f44336;'>Error: {agent['error'][:100]}</div>\n"
            
            html += "            </div>\n"
        
        html += """
        </div>
        
        <h2>📈 Decisions Summary</h2>
"""
        
        # 决策汇总
        if trace.decisions:
            action_counts = defaultdict(int)
            for decision in trace.decisions:
                action_counts[decision['action']] += 1
            
            html += f"        <p>Total Decisions: {len(trace.decisions)}</p>\n"
            html += "        <ul class='decision-list'>\n"
            
            for action, count in sorted(action_counts.items()):
                html += f"            <li><span class='badge {action}'>{action}</span>: {count}</li>\n"
            
            html += "        </ul>\n"
        
        # 错误列表
        if trace.errors:
            html += """
        <h2>⚠️ Errors & Warnings</h2>
        <div class="error-list">
"""
            for error in trace.errors[:20]:
                html += f"            <p><strong>{error['time']}</strong> [{error['level']}]: {error['message']}</p>\n"
            
            html += "        </div>\n"
        
        html += """
    </div>
</body>
</html>
"""
        return html
    
    def visualize(self, run_id: Optional[str] = None, date: Optional[str] = None, 
                  output_html: Optional[str] = None):
        """可视化执行追踪"""
        traces = self.collect_trace(run_id, date)
        
        if not traces:
            print("⚠️  No traces found.")
            return
        
        print(f"📊 Found {len(traces)} execution trace(s)")
        
        for trace in traces:
            if output_html:
                # 生成 HTML
                html = self.generate_html_trace(trace)
                
                output_path = Path(output_html)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                
                print(f"✅ HTML trace saved to: {output_path}")
            else:
                # 打印文本追踪
                print(self.format_text_trace(trace))


def main():
    parser = argparse.ArgumentParser(
        description='Visualize execution traces from logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trace specific run_id
  %(prog)s --run-id backtest_20251215_001
  
  # Trace all runs on a date
  %(prog)s --date 2025-12-15
  
  # Generate HTML visualization
  %(prog)s --run-id backtest_20251215_001 --html trace.html
        """
    )
    
    parser.add_argument('--log-dir', default='logs/stockbench', help='Log directory')
    parser.add_argument('--run-id', help='Run ID to trace')
    parser.add_argument('--date', help='Date (YYYY-MM-DD)')
    parser.add_argument('--html', help='Output HTML file')
    
    args = parser.parse_args()
    
    if not args.run_id and not args.date:
        parser.error("Must specify either --run-id or --date")
    
    # 创建可视化器
    visualizer = TraceVisualizer(args.log_dir)
    
    # 可视化追踪
    visualizer.visualize(args.run_id, args.date, args.html)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
