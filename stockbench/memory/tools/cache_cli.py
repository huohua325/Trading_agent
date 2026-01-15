"""
统一缓存管理CLI - 使用融合后的CacheStore

使用方法:
    python -m stockbench.memory.tools.cache_cli stats
    python -m stockbench.memory.tools.cache_cli health
    python -m stockbench.memory.tools.cache_cli cleanup llm --days 30
"""

import sys
import argparse
from pathlib import Path
import json
from loguru import logger

# 导入融合后的系统
from ..store import MemoryStore
from ..layers.cache_tools import CacheCleanupTool


def setup_logger():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )


def print_json(data: dict, indent: int = 2):
    """格式化打印JSON"""
    print(json.dumps(data, indent=indent, ensure_ascii=False))


def get_cache_store():
    """获取CacheStore实例"""
    # 创建MemoryStore，它包含增强后的CacheStore
    memory_store = MemoryStore(base_path="storage")
    return memory_store.cache


def cmd_stats(args):
    """查看缓存统计"""
    cache = get_cache_store()
    stats = cache.get_cache_stats()
    
    print("\n[*] Cache Statistics\n" + "=" * 60)
    
    total_files = sum(ns["file_count"] for ns in stats.values())
    total_size = sum(ns["total_size_mb"] for ns in stats.values())
    
    print(f"Total Namespaces: {len(stats)}")
    print(f"Total Files: {total_files:,}")
    print(f"Total Size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    print("\n" + "-" * 60)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total_size_mb"], reverse=True)
    
    print(f"\n{'Namespace':<20} {'Files':>10} {'Size (MB)':>12} {'Subdirs':>10}")
    print("-" * 60)
    
    for ns, info in sorted_stats:
        print(f"{ns:<20} {info['file_count']:>10,} {info['total_size_mb']:>12.2f} {len(info['subdirs']):>10}")
    
    if args.json:
        print("\n" + "=" * 60)
        print_json(stats)


def cmd_health(args):
    """健康检查"""
    cache = get_cache_store()
    cleanup_tool = CacheCleanupTool(cache)
    
    report = cleanup_tool.analyze_cache_health()
    
    print("\n[*] Cache Health Report\n" + "=" * 60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Health Score: {report['health_score'].upper()}")
    print(f"\nTotal Namespaces: {report['summary']['total_namespaces']}")
    print(f"Total Files: {report['summary']['total_files']:,}")
    print(f"Total Size: {report['summary']['total_size_mb']:.2f} MB ({report['summary']['total_size_gb']:.2f} GB)")
    
    if report['warnings']:
        print(f"\n[!] Warnings ({len(report['warnings'])})")
        print("-" * 60)
        for i, warning in enumerate(report['warnings'], 1):
            print(f"{i}. [{warning['type']}] {warning['message']}")
    else:
        print("\n[+] No warnings - cache is healthy!")
    
    if args.json:
        print("\n" + "=" * 60)
        print_json(report)


def cmd_list(args):
    """列出所有命名空间"""
    cache = get_cache_store()
    namespaces = cache.discover_namespaces()
    
    print("\n[*] Cache Namespaces\n" + "=" * 60)
    for ns in namespaces:
        ns_path = cache.get_namespace_path(ns)
        print(f"  • {ns:<20} ({ns_path})")
    
    print(f"\nTotal: {len(namespaces)} namespaces")


def cmd_cleanup(args):
    """清理缓存"""
    cache = get_cache_store()
    cleanup_tool = CacheCleanupTool(cache)
    
    namespace = args.namespace
    days_old = args.days
    dry_run = not args.execute
    
    if dry_run:
        print("\n[*] Cache Cleanup (DRY RUN)\n" + "=" * 60)
        print("[!] This is a dry run - no files will be deleted")
        print("    Use --execute to actually delete files\n")
    else:
        print("\n[*] Cache Cleanup (EXECUTING)\n" + "=" * 60)
    
    result = cleanup_tool.cleanup_old_files(namespace, days_old, dry_run)
    
    print(f"Namespace: {namespace}")
    print(f"Age Threshold: {days_old} days")
    print(f"Files Found: {result['file_count']:,}")
    print(f"Total Size: {result['size_mb']:.2f} MB")
    
    if result['files']:
        print(f"\nSample files (showing first 10):")
        for f in result['files']:
            print(f"  • {f['path']} ({f['size_kb']:.2f} KB, {f['age_days']} days old)")
    
    if args.json:
        print("\n" + "=" * 60)
        print_json(result)


def cmd_validate(args):
    """验证缓存完整性"""
    cache = get_cache_store()
    cleanup_tool = CacheCleanupTool(cache)
    
    namespace = args.namespace
    
    print(f"\n[*] Validating {namespace} cache...\n")
    
    result = cleanup_tool.validate_cache_integrity(namespace)
    
    print("=" * 60)
    print(f"Valid Files: {result['valid_files']:,}")
    print(f"Corrupted Files: {result['corrupted_files']:,}")
    print(f"Integrity Score: {result['integrity_score']:.2f}%")
    
    if result['corrupted_list']:
        print(f"\n[!] Corrupted files found:")
        for f in result['corrupted_list']:
            print(f"  • {f['path']}")
            print(f"    Error: {f['error']}")
    
    if args.json:
        print("\n" + "=" * 60)
        print_json(result)


def main():
    """主函数"""
    setup_logger()
    
    parser = argparse.ArgumentParser(
        description="Unified Cache Management CLI (Memory-integrated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # 各种命令
    subparsers.add_parser('stats', help='Show cache statistics')
    subparsers.add_parser('health', help='Run cache health check')
    subparsers.add_parser('list', help='List all cache namespaces')
    
    parser_cleanup = subparsers.add_parser('cleanup', help='Clean up old cache files')
    parser_cleanup.add_argument('namespace', help='Namespace to clean')
    parser_cleanup.add_argument('--days', type=int, default=30, help='Delete files older than N days')
    parser_cleanup.add_argument('--execute', action='store_true', help='Actually delete files')
    
    parser_validate = subparsers.add_parser('validate', help='Validate cache integrity')
    parser_validate.add_argument('namespace', help='Namespace to validate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        'stats': cmd_stats,
        'health': cmd_health,
        'list': cmd_list,
        'cleanup': cmd_cleanup,
        'validate': cmd_validate,
    }
    
    if args.command in commands:
        try:
            commands[args.command](args)
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if args.json:
                print_json({"error": str(e)})
            sys.exit(1)


if __name__ == '__main__':
    main()
