# StockBench 文档索引

> **更新日期**: 2025-12-16  
> **文档组织**: 分类清晰，便于查找

---

## 📚 文档导航

### 1️⃣ 使用指南 (`guides/`)

核心使用文档，适合所有用户阅读：

| 文档 | 说明 | 适用人群 |
|------|------|---------|
| **SYSTEM_UPGRADE_GUIDE.md** | 系统升级完整指南 | 所有用户 ⭐⭐⭐ |
| **MIGRATION_GUIDE.md** | v0.8 → v1.0 迁移指南 | 升级用户 |
| **STRUCTURED_LOGGING_MIGRATION.md** | 结构化日志迁移指南 | 开发者 |
| **FUNCTION_CALLING_GUIDE.md** | Function Calling 使用指南 | Agent 开发者 |
| **STOCKBENCH_LEARNING_GUIDE.md** | StockBench 学习指南 | 新用户 |

### 2️⃣ 架构文档 (`architecture/`)

系统架构与设计文档：

| 文档 | 说明 |
|------|------|
| **PROJECT_STRUCTURE.md** | 项目结构详解 |

### 3️⃣ 规划文档 (`planning/`)

未来规划与升级计划：

| 文档 | 说明 |
|------|------|
| **UPGRADE_PLAN_PHASE2.md** | 框架升级计划与展望（最新） |
| `archive/UPGRADE_PLAN_PHASE2_OLD.md` | 历史版本（已归档） |

### 4️⃣ 日志系统 (`logging/`)

日志系统优化相关文档：

| 文档 | 说明 |
|------|------|
| **LOGGING_OPTIMIZATION_IMPLEMENTATION.md** | 日志优化实施报告（Phase 1-6） |
| **LOGGING_OPTIMIZATION_PLAN.md** | 日志优化完整计划 |
| **LOG_ANALYSIS_TOOLS.md** | 日志分析工具使用手册 |
| **SYSTEM_UPGRADE_GUIDE_LOGGING.md** | 日志系统优化详细说明 |

### 5️⃣ 构建方法 (`ConstructMethod/`)

特征构建相关文档（历史）

### 6️⃣ Memory 系统 (`MemoryMethod/`)

Memory 系统相关文档（历史）

---

## 🚀 快速开始

**新用户**:
1. 阅读 `guides/STOCKBENCH_LEARNING_GUIDE.md`
2. 参考 `guides/SYSTEM_UPGRADE_GUIDE.md`

**升级用户**:
1. 阅读 `guides/MIGRATION_GUIDE.md`
2. 查看 `guides/SYSTEM_UPGRADE_GUIDE.md` 了解新特性

**开发者**:
1. 查看 `architecture/PROJECT_STRUCTURE.md` 了解架构
2. 参考 `planning/UPGRADE_PLAN_PHASE2.md` 了解未来方向
3. 使用 `logging/LOG_ANALYSIS_TOOLS.md` 进行日志分析

---

## 📊 文档地图

```
docs/
├── README.md                          # 本文档（索引）
│
├── guides/                            # 📖 使用指南
│   ├── SYSTEM_UPGRADE_GUIDE.md        # ⭐ 系统升级完整指南
│   ├── MIGRATION_GUIDE.md             # 迁移指南
│   ├── STRUCTURED_LOGGING_MIGRATION.md # 日志迁移指南
│   ├── FUNCTION_CALLING_GUIDE.md      # Function Calling 指南
│   └── STOCKBENCH_LEARNING_GUIDE.md   # 学习指南
│
├── architecture/                      # 🏗️ 架构文档
│   └── PROJECT_STRUCTURE.md           # 项目结构
│
├── planning/                          # 📋 规划文档
│   ├── UPGRADE_PLAN_PHASE2.md         # 升级计划（最新）
│   └── archive/                       # 历史归档
│       └── UPGRADE_PLAN_PHASE2_OLD.md
│
├── logging/                           # 📊 日志系统
│   ├── LOGGING_OPTIMIZATION_IMPLEMENTATION.md  # 实施报告
│   ├── LOGGING_OPTIMIZATION_PLAN.md            # 优化计划
│   ├── LOG_ANALYSIS_TOOLS.md                   # 工具手册
│   └── SYSTEM_UPGRADE_GUIDE_LOGGING.md         # 详细说明
│
├── ConstructMethod/                   # 🔧 构建方法（历史）
└── MemoryMethod/                      # 💾 Memory 系统（历史）
```

---

## 🎯 推荐阅读路径

### 路径 1: 快速上手（新用户）
1. `guides/STOCKBENCH_LEARNING_GUIDE.md`
2. `guides/SYSTEM_UPGRADE_GUIDE.md` - 第 1-3 章
3. 根据需求查阅其他文档

### 路径 2: 系统升级（升级用户）
1. `guides/MIGRATION_GUIDE.md`
2. `guides/SYSTEM_UPGRADE_GUIDE.md`
3. `logging/LOGGING_OPTIMIZATION_IMPLEMENTATION.md`

### 路径 3: 深度开发（开发者）
1. `architecture/PROJECT_STRUCTURE.md`
2. `guides/SYSTEM_UPGRADE_GUIDE.md`
3. `planning/UPGRADE_PLAN_PHASE2.md`
4. `logging/LOG_ANALYSIS_TOOLS.md`

### 路径 4: 日志优化（运维）
1. `logging/LOGGING_OPTIMIZATION_IMPLEMENTATION.md`
2. `logging/LOG_ANALYSIS_TOOLS.md`
3. `guides/STRUCTURED_LOGGING_MIGRATION.md`

---

## 📝 文档更新记录

| 日期 | 更新内容 |
|------|---------|
| 2025-12-16 | 重组文档结构，创建分类目录 |
| 2025-12-16 | 完成 Phase 6 日志分析工具 |
| 2025-12-16 | 更新 UPGRADE_PLAN_PHASE2.md（框架分析与展望） |
| 2025-12-15 | 完成 Phase 1-5 日志优化 |
| 2025-12-14 | 完成 Memory & Message 系统 |

---

*文档维护: StockBench Team*
