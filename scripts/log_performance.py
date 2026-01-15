#!/usr/bin/env python3
"""
Log Performance Analyzer - 日志性能分析工具

分析日志中的性能指标，生成统计报告。

Usage:
    # 分析今天的日志
    python scripts/log_performance.py
    
    # 分析特定日期
    python scripts/log_performance.py --date 2025-12-15
    
    # 生成详细报告
    python scripts/log_performance.py --detailed --output report.txt
    
    # 只分析 LLM 性能
    python scripts/log_performance.py --focus llm
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
import statistics


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, log_dir: str = "logs/stockbench"):
        self.log_dir = Path(log_dir)
        self.metrics = {
            'agents': defaultdict(list),
            'llm_calls': defaultdict(list),
            'data_fetches': defaultdict(list),
            'decisions': defaultdict(list),
            'orders': defaultdict(list),
            'features': defaultdict(list),
        }
    
    def find_log_files(self, date: Optional[str] = None) -> List[Path]:
        """查找日志文件"""
        if not self.log_dir.exists():
            print(f"❌ Log directory not found: {self.log_dir}")
            return []
        
        if date:
            pattern = f"{date}.log"
        else:
            # 默认今天
            today = datetime.now().strftime("%Y-%m-%d")
            pattern = f"{today}.log"
        
        files = sorted(self.log_dir.glob(pattern))
        return files
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析日志行"""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return None
    
    def collect_metrics(self, date: Optional[str] = None):
        """收集性能指标"""
        log_files = self.find_log_files(date)
        if not log_files:
            print(f"⚠️  No log files found for date: {date or 'today'}")
            return
        
        print(f"📊 Analyzing {len(log_files)} log file(s)...")
        
        for log_file in log_files:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = self.parse_log_line(line)
                    if not entry:
                        continue
                    
                    # Agent 性能
                    if entry.get('agent_name') and entry.get('duration_ms'):
                        agent_name = entry['agent_name']
                        duration = entry['duration_ms']
                        status = entry.get('status', 'unknown')
                        
                        self.metrics['agents'][agent_name].append({
                            'duration_ms': duration,
                            'status': status,
                            'input_count': entry.get('input_count'),
                            'output_count': entry.get('output_count'),
                        })
                    
                    # LLM 性能
                    if 'LLM_CALL' in entry.get('message', '') or 'LLM_CACHE' in entry.get('message', ''):
                        model = entry.get('model', 'unknown')
                        self.metrics['llm_calls'][model].append({
                            'latency_ms': entry.get('latency_ms'),
                            'total_tokens': entry.get('total_tokens'),
                            'prompt_tokens': entry.get('prompt_tokens'),
                            'completion_tokens': entry.get('completion_tokens'),
                            'cache_hit': entry.get('cache_hit', False),
                            'status': entry.get('status', 'unknown'),
                            'estimated_cost': entry.get('estimated_cost', 0),
                        })
                    
                    # 数据获取性能
                    if 'DATA_FETCH' in entry.get('message', '') or 'DATA_CACHE' in entry.get('message', ''):
                        data_type = entry.get('data_type', 'unknown')
                        self.metrics['data_fetches'][data_type].append({
                            'fetch_time_ms': entry.get('fetch_time_ms'),
                            'records_fetched': entry.get('records_fetched'),
                            'cache_hit': entry.get('cache_hit', False),
                            'status': entry.get('status', 'unknown'),
                        })
                    
                    # 决策性能
                    if 'AGENT_DECISION' in entry.get('message', ''):
                        symbol = entry.get('symbol', 'unknown')
                        self.metrics['decisions'][symbol].append({
                            'decision_time_ms': entry.get('decision_time_ms'),
                            'confidence': entry.get('confidence'),
                            'action': entry.get('action'),
                        })
                    
                    # 订单性能
                    if 'BT_ORDER' in entry.get('message', ''):
                        symbol = entry.get('symbol', 'unknown')
                        self.metrics['orders'][symbol].append({
                            'status': entry.get('status'),
                            'qty': entry.get('qty'),
                            'commission': entry.get('commission'),
                        })
                    
                    # 特征构建性能
                    if 'FEATURE_BUILD' in entry.get('message', ''):
                        feature_type = entry.get('feature_type', 'unknown')
                        self.metrics['features'][feature_type].append({
                            'construction_time_ms': entry.get('construction_time_ms'),
                            'data_points': entry.get('data_points'),
                            'quality_score': entry.get('quality_score'),
                        })
    
    def analyze_agents(self) -> Dict[str, Any]:
        """分析 Agent 性能"""
        if not self.metrics['agents']:
            return {}
        
        report = {}
        
        for agent_name, records in self.metrics['agents'].items():
            durations = [r['duration_ms'] for r in records if r['duration_ms']]
            
            if not durations:
                continue
            
            success_count = sum(1 for r in records if r.get('status') == 'success')
            failed_count = sum(1 for r in records if r.get('status') == 'failed')
            
            report[agent_name] = {
                'total_executions': len(records),
                'success_count': success_count,
                'failed_count': failed_count,
                'success_rate': success_count / len(records) if records else 0,
                'avg_duration_ms': statistics.mean(durations),
                'median_duration_ms': statistics.median(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'total_duration_ms': sum(durations),
            }
        
        return report
    
    def analyze_llm(self) -> Dict[str, Any]:
        """分析 LLM 性能"""
        if not self.metrics['llm_calls']:
            return {}
        
        report = {}
        
        for model, records in self.metrics['llm_calls'].items():
            latencies = [r['latency_ms'] for r in records if r.get('latency_ms')]
            tokens = [r['total_tokens'] for r in records if r.get('total_tokens')]
            costs = [r['estimated_cost'] for r in records if r.get('estimated_cost')]
            
            cache_hits = sum(1 for r in records if r.get('cache_hit'))
            
            report[model] = {
                'total_calls': len(records),
                'cache_hits': cache_hits,
                'cache_hit_rate': cache_hits / len(records) if records else 0,
                'avg_latency_ms': statistics.mean(latencies) if latencies else 0,
                'median_latency_ms': statistics.median(latencies) if latencies else 0,
                'total_tokens': sum(tokens),
                'avg_tokens': statistics.mean(tokens) if tokens else 0,
                'total_cost': sum(costs),
                'avg_cost': statistics.mean(costs) if costs else 0,
            }
        
        return report
    
    def analyze_data_fetches(self) -> Dict[str, Any]:
        """分析数据获取性能"""
        if not self.metrics['data_fetches']:
            return {}
        
        report = {}
        
        for data_type, records in self.metrics['data_fetches'].items():
            fetch_times = [r['fetch_time_ms'] for r in records if r.get('fetch_time_ms')]
            cache_hits = sum(1 for r in records if r.get('cache_hit'))
            
            report[data_type] = {
                'total_fetches': len(records),
                'cache_hits': cache_hits,
                'cache_hit_rate': cache_hits / len(records) if records else 0,
                'avg_fetch_time_ms': statistics.mean(fetch_times) if fetch_times else 0,
                'total_records': sum(r.get('records_fetched', 0) for r in records),
            }
        
        return report
    
    def analyze_decisions(self) -> Dict[str, Any]:
        """分析决策统计"""
        if not self.metrics['decisions']:
            return {}
        
        all_records = []
        for records in self.metrics['decisions'].values():
            all_records.extend(records)
        
        confidences = [r['confidence'] for r in all_records if r.get('confidence')]
        actions = [r['action'] for r in all_records if r.get('action')]
        
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action] += 1
        
        return {
            'total_decisions': len(all_records),
            'avg_confidence': statistics.mean(confidences) if confidences else 0,
            'action_distribution': dict(action_counts),
        }
    
    def generate_report(self, detailed: bool = False) -> str:
        """生成性能报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("📊 LOG PERFORMANCE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Agent 性能
        agent_report = self.analyze_agents()
        if agent_report:
            lines.append("🤖 AGENT PERFORMANCE")
            lines.append("-" * 80)
            
            for agent_name, stats in sorted(agent_report.items()):
                lines.append(f"\n[{agent_name}]")
                lines.append(f"  Executions: {stats['total_executions']} (✅ {stats['success_count']} / ❌ {stats['failed_count']})")
                lines.append(f"  Success Rate: {stats['success_rate']:.1%}")
                lines.append(f"  Duration: avg={stats['avg_duration_ms']:.1f}ms, median={stats['median_duration_ms']:.1f}ms")
                lines.append(f"  Range: {stats['min_duration_ms']:.1f}ms - {stats['max_duration_ms']:.1f}ms")
                lines.append(f"  Total Time: {stats['total_duration_ms']:.1f}ms")
            
            lines.append("")
        
        # LLM 性能
        llm_report = self.analyze_llm()
        if llm_report:
            lines.append("🧠 LLM PERFORMANCE")
            lines.append("-" * 80)
            
            for model, stats in sorted(llm_report.items()):
                lines.append(f"\n[{model}]")
                lines.append(f"  Total Calls: {stats['total_calls']}")
                lines.append(f"  Cache Hits: {stats['cache_hits']} ({stats['cache_hit_rate']:.1%})")
                lines.append(f"  Latency: avg={stats['avg_latency_ms']:.1f}ms, median={stats['median_latency_ms']:.1f}ms")
                lines.append(f"  Tokens: total={stats['total_tokens']:,}, avg={stats['avg_tokens']:.0f}")
                lines.append(f"  Cost: total=${stats['total_cost']:.4f}, avg=${stats['avg_cost']:.4f}")
            
            lines.append("")
        
        # 数据获取性能
        data_report = self.analyze_data_fetches()
        if data_report:
            lines.append("📦 DATA FETCH PERFORMANCE")
            lines.append("-" * 80)
            
            for data_type, stats in sorted(data_report.items()):
                lines.append(f"\n[{data_type}]")
                lines.append(f"  Total Fetches: {stats['total_fetches']}")
                lines.append(f"  Cache Hits: {stats['cache_hits']} ({stats['cache_hit_rate']:.1%})")
                lines.append(f"  Avg Fetch Time: {stats['avg_fetch_time_ms']:.1f}ms")
                lines.append(f"  Total Records: {stats['total_records']:,}")
            
            lines.append("")
        
        # 决策统计
        decision_report = self.analyze_decisions()
        if decision_report:
            lines.append("📈 DECISION STATISTICS")
            lines.append("-" * 80)
            lines.append(f"  Total Decisions: {decision_report['total_decisions']}")
            lines.append(f"  Avg Confidence: {decision_report['avg_confidence']:.2%}")
            lines.append(f"  Action Distribution:")
            for action, count in sorted(decision_report['action_distribution'].items()):
                pct = count / decision_report['total_decisions']
                lines.append(f"    - {action}: {count} ({pct:.1%})")
            
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_report(self, output_file: str, detailed: bool = False):
        """保存报告到文件"""
        report = self.generate_report(detailed)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze log performance metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze today's logs
  %(prog)s
  
  # Analyze specific date
  %(prog)s --date 2025-12-15
  
  # Generate detailed report
  %(prog)s --detailed
  
  # Save to file
  %(prog)s --output performance_report.txt
  
  # Focus on specific metrics
  %(prog)s --focus llm
        """
    )
    
    parser.add_argument('--log-dir', default='logs/stockbench', help='Log directory')
    parser.add_argument('--date', help='Date (YYYY-MM-DD)')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed report')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--focus', choices=['agents', 'llm', 'data', 'decisions'], 
                       help='Focus on specific metrics')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = PerformanceAnalyzer(args.log_dir)
    
    # 收集指标
    analyzer.collect_metrics(args.date)
    
    # 生成报告
    if args.output:
        analyzer.save_report(args.output, args.detailed)
    else:
        print(analyzer.generate_report(args.detailed))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
