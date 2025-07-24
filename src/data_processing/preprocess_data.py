"""
Amazon Electronics 数据预处理模块
处理评分数据和商品元数据，为推荐系统建模做准备
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_dir="../../data"):
        self.data_dir = data_dir
        self.ratings_df = None
        self.metadata_df = None
        self.processed_ratings = None
        self.user_item_matrix = None
        
    def load_data(self):
        """加载原始数据"""
        print("📂 加载数据...")
        
        # 加载评分数据
        ratings_file = f"{self.data_dir}/ratings_Electronics.csv"
        self.ratings_df = pd.read_csv(
            ratings_file, 
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        # 加载商品元数据
        metadata_file = f"{self.data_dir}/meta_Electronics.json"
        metadata_list = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = eval(line.strip())
                    metadata_list.append({
                        'item_id': item.get('asin', ''),
                        'title': item.get('title', ''),
                        'categories': item.get('categories', []),
                        'price': item.get('price', None),
                        'brand': item.get('brand', ''),
                        'description': item.get('description', '')
                    })
                except:
                    continue
        
        self.metadata_df = pd.DataFrame(metadata_list)
        
        print(f"✅ 数据加载完成:")
        print(f"   - 评分数据: {len(self.ratings_df):,} 条")
        print(f"   - 商品元数据: {len(self.metadata_df):,} 个")
        
    def clean_ratings_data(self):
        """清洗评分数据"""
        print("\n🧹 清洗评分数据...")
        
        original_size = len(self.ratings_df)
        
        # 移除重复评分（同一用户对同一商品的多次评分，保留最新的）
        self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
        self.ratings_df = self.ratings_df.sort_values('timestamp').drop_duplicates(
            subset=['user_id', 'item_id'], keep='last'
        )
        
        # 过滤评分数据：保留评分次数>=5的用户和被评分次数>=5的商品
        user_counts = self.ratings_df['user_id'].value_counts()
        item_counts = self.ratings_df['item_id'].value_counts()
        
        active_users = user_counts[user_counts >= 5].index
        popular_items = item_counts[item_counts >= 5].index
        
        self.processed_ratings = self.ratings_df[
            (self.ratings_df['user_id'].isin(active_users)) &
            (self.ratings_df['item_id'].isin(popular_items))
        ].copy()
        
        print(f"   - 原始评分数: {original_size:,}")
        print(f"   - 清洗后评分数: {len(self.processed_ratings):,}")
        print(f"   - 活跃用户数: {self.processed_ratings['user_id'].nunique():,}")
        print(f"   - 热门商品数: {self.processed_ratings['item_id'].nunique():,}")
        
    def create_user_item_matrix(self):
        """创建用户-物品评分矩阵"""
        print("\n📊 创建用户-物品矩阵...")
        
        # 创建用户和物品的ID映射
        unique_users = self.processed_ratings['user_id'].unique()
        unique_items = self.processed_ratings['item_id'].unique()
        
        self.user_id_map = {user: idx for idx, user in enumerate(unique_users)}
        self.item_id_map = {item: idx for idx, item in enumerate(unique_items)}
        
        # 反向映射
        self.idx_to_user = {idx: user for user, idx in self.user_id_map.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_id_map.items()}
        
        # 添加映射后的ID
        self.processed_ratings['user_idx'] = self.processed_ratings['user_id'].map(self.user_id_map)
        self.processed_ratings['item_idx'] = self.processed_ratings['item_id'].map(self.item_id_map)
        
        # 创建稀疏矩阵
        from scipy.sparse import csr_matrix
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.user_item_matrix = csr_matrix(
            (self.processed_ratings['rating'], 
             (self.processed_ratings['user_idx'], self.processed_ratings['item_idx'])),
            shape=(n_users, n_items)
        )
        
        # 计算稀疏度
        sparsity = 1 - (len(self.processed_ratings) / (n_users * n_items))
        
        print(f"   - 矩阵维度: {n_users:,} x {n_items:,}")
        print(f"   - 稀疏度: {sparsity:.4f} ({sparsity*100:.2f}%)")
        
    def analyze_data_distribution(self):
        """分析数据分布"""
        print("\n📈 数据分布分析...")
        
        # 评分分布
        rating_dist = self.processed_ratings['rating'].value_counts().sort_index()
        print(f"   - 评分分布:")
        for rating, count in rating_dist.items():
            print(f"     {rating}星: {count:,} ({count/len(self.processed_ratings)*100:.1f}%)")
        
        # 用户活跃度分布
        user_activity = self.processed_ratings['user_id'].value_counts()
        print(f"   - 用户活跃度:")
        print(f"     平均评分数: {user_activity.mean():.1f}")
        print(f"     中位数评分数: {user_activity.median():.1f}")
        print(f"     最大评分数: {user_activity.max()}")
        
        # 商品热门度分布
        item_popularity = self.processed_ratings['item_id'].value_counts()
        print(f"   - 商品热门度:")
        print(f"     平均被评分数: {item_popularity.mean():.1f}")
        print(f"     中位数被评分数: {item_popularity.median():.1f}")
        print(f"     最大被评分数: {item_popularity.max()}")
        
    def save_processed_data(self):
        """保存处理后的数据"""
        print("\n💾 保存处理后的数据...")
        
        # 保存清洗后的评分数据
        self.processed_ratings.to_csv(f"{self.data_dir}/processed_ratings.csv", index=False)
        
        # 保存ID映射
        import pickle
        with open(f"{self.data_dir}/user_id_map.pkl", 'wb') as f:
            pickle.dump(self.user_id_map, f)
        with open(f"{self.data_dir}/item_id_map.pkl", 'wb') as f:
            pickle.dump(self.item_id_map, f)
        
        # 保存用户-物品矩阵
        from scipy.sparse import save_npz
        save_npz(f"{self.data_dir}/user_item_matrix.npz", self.user_item_matrix)
        
        print("✅ 数据保存完成!")
        
    def run_preprocessing(self):
        """运行完整的数据预处理流程"""
        print("🚀 开始数据预处理...\n")
        
        self.load_data()
        self.clean_ratings_data()
        self.create_user_item_matrix()
        self.analyze_data_distribution()
        self.save_processed_data()
        
        print(f"\n🎉 数据预处理完成!")
        print(f"   - 最终数据集: {len(self.processed_ratings):,} 条评分")
        print(f"   - 用户数: {len(self.user_id_map):,}")
        print(f"   - 商品数: {len(self.item_id_map):,}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_preprocessing()
