# 电商智能推荐系统

## 🎯 项目概述
基于Amazon Electronics数据集构建的完整推荐系统，实现了协同过滤、基于内容推荐和混合推荐三种算法。

## 📊 项目成果
-  **数据规模**: 处理210万条评分数据，25万用户，14万商品
-  **算法实现**: SVD协同过滤、1900+维内容推荐、混合推荐系统
-  **性能指标**: RMSE 1.28, MAE 0.99, 特征维度1900+
-  **工程完整**: 数据预处理、模型训练、推荐生成完整流程

## 🧠 核心算法

### 1. 协同过滤 (SVD矩阵分解)
- **性能**: RMSE 1.28, MAE 0.99
- **特点**: 发现潜在兴趣，个性化强
- **文件**: `src/models/collaborative_filtering.py`

### 2. 基于内容推荐
- **特征**: 1900+维多模态特征(类别、品牌、价格、文本)
- **特点**: 可解释性强，解决冷启动问题
- **文件**: `src/models/content_based_recommendation.py`

### 3. 混合推荐系统
- **策略**: 加权组合、智能切换、混合策略
- **特点**: 融合两者优势，智能策略选择
- **文件**: `src/models/hybrid_recommendation.py`

## 🔍 详细分析

### 用户行为分析

- **评分分布**: 59%为5星，79.4%为4-5星 (正向偏好)
- **用户活跃度**: 平均8.3个评分，中位数6个
- **商品热门度**: 典型长尾分布，中位数4个评分

### 推荐效果分析

- **个性化程度**: 不同用户推荐结果差异明显
- **类别相关性**: 推荐商品与用户历史高度匹配
- **价格合理性**: 推荐价格符合用户消费水平
- **品牌一致性**: 推荐知名品牌，质量可靠

### 算法对比分析

| 维度     | 协同过滤 | 基于内容 | 混合推荐 |
| -------- | -------- | -------- | -------- |
| 个性化   | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐    |
| 冷启动   | ❌        | ✅        | ✅        |
| 可解释性 | ⭐⭐       | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐     |
| 多样性   | ⭐⭐⭐⭐     | ⭐⭐⭐      | ⭐⭐⭐⭐⭐    |
| 鲁棒性   | ⭐⭐⭐      | ⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐    |

## 📁 项目结构

```
recommendation_project/
├── data/                    # 数据文件
│   ├── ratings_Electronics.csv      # 原始评分数据
│   ├── meta_Electronics.json        # 商品元数据
│   ├── processed_ratings.csv        # 清洗后评分数据
│   ├── item_features.csv           # 商品特征矩阵
│   ├── svd_model.pkl               # 协同过滤模型
│   └── feature_matrix.npy          # 标准化特征矩阵
├── src/
│   ├── data_processing/
│   │   ├── download_data.py         # 数据下载脚本
│   │   └── preprocess_data.py       # 数据预处理
│   └── models/
│       ├── collaborative_filtering.py    # 协同过滤模型
│       ├── content_based_recommendation.py # 基于内容推荐
│       ├── hybrid_recommendation.py      # 混合推荐系统
├── test.py                  # 功能测试脚本
├── requirements.txt         # 依赖包
├── README.md               # 项目说明
└── PROJECT_SUMMARY.md      # 项目详细总结
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境
conda create -n recommendation python=3.8
conda activate recommendation

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

⚠️ **重要说明**: 由于GitHub文件大小限制，大型数据文件未包含在仓库中。请按以下步骤获取数据：

#### 方法1: 自动下载和处理 (推荐)
```bash
cd src/data_processing
python download_data.py    # 下载Amazon Electronics数据集
python preprocess_data.py  # 数据清洗和预处理
```

#### 方法2: 手动下载
1. 从 [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/) 下载以下文件：
   - `ratings_Electronics.csv` (评分数据)
   - `meta_Electronics.json` (商品元数据)
2. 将文件放置在 `data/` 目录下
3. 运行预处理脚本：
```bash
cd src/data_processing
python preprocess_data.py
```

#### 生成的文件
运行上述步骤后，将在 `data/` 目录下生成以下文件：
- `processed_ratings.csv` (210万条预处理评分数据)
- `feature_matrix.npy` (商品特征矩阵, ~2GB)
- `svd_model.pkl` (训练好的SVD模型)
- `item_features.csv` (商品特征表)
- `tfidf_vectorizer.pkl` (TF-IDF向量化器)
- `price_scaler.pkl` (价格标准化器)

### 3. 模型训练
```bash
cd src/models
python collaborative_filtering.py      # 训练SVD协同过滤模型
python content_based_recommendation.py # 训练基于内容推荐模型
python hybrid_recommendation.py        # 训练混合推荐系统
```

### 4. 功能测试
```bash
# 运行完整功能测试
python test.py
```

### 5. 使用推荐系统
```python
# 示例：使用混合推荐系统
import sys
sys.path.append('src/models')
from hybrid_recommendation import HybridRecommendationSystem

# 初始化推荐系统
hybrid_system = HybridRecommendationSystem('data')
hybrid_system.load_models()

# 为用户生成推荐
user_id = 'ADLVFFE4VBT8'  # 示例用户ID
recommendations = hybrid_system.get_hybrid_recommendations(user_id, 'weighted', 5)

# 查看推荐结果
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. 商品: {rec['item_id']} | 评分: {rec['hybrid_score']:.3f}")
```

## 📈 性能指标

| 算法 | RMSE | MAE | 特点 |
|------|------|-----|------|
| 协同过滤 | 1.28 | 0.99 | 发现潜在兴趣，个性化强 |
| 基于内容 | - | - | 解决冷启动，可解释性强 |
| 混合推荐 | 融合优势 | 融合优势 | 结合两者优点，鲁棒性强 |

## 🎯 技术亮点
1. **大规模数据处理**: 高效处理210万条评分数据
2. **多模态特征工程**: 融合类别、品牌、价格、文本等1900+维特征
3. **算法融合**: 协同过滤与内容推荐的智能混合
4. **内存优化**: 解决大规模相似度矩阵的内存问题
5. **工程完整性**: 从数据处理到模型部署的完整流程

## 📊 数据集信息
- **数据来源**: Amazon Electronics Dataset
- **评分数据**: 2,109,869 条用户-商品评分
- **用户数**: 253,994 个活跃用户
- **商品数**: 145,199 个电子产品
- **特征维度**: 1900+ 维多模态特征
- **数据稀疏度**: 99.99% (典型推荐系统挑战)

## ✨ 主要特性

-  **多算法实现**: 协同过滤、内容推荐、混合推荐
- **特征工程**: TF-IDF文本特征、One-hot类别编码、价格标准化
-  **模型持久化**: 支持模型保存和快速加载
-  **性能优化**: 矩阵运算优化，支持大规模数据
- **完整测试**: 功能完整性测试脚本

## 🔧 技术栈

- **核心语言**: Python 3.8+
- **数据处理**: Pandas, NumPy, Scipy
- **机器学习**: scikit-learn, scikit-surprise
- **特征工程**: TF-IDF向量化, One-hot编码, 标准化
- **可视化**: Matplotlib, Seaborn
- **模型持久化**: Pickle, NumPy
