"""
基于内容的推荐系统
利用商品特征（类别、品牌、价格等）计算商品相似度，生成推荐
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.ratings_df = None
        self.metadata_df = None
        self.item_features = None
        self.item_similarity_matrix = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载数据"""
        print("📂 加载数据...")
        
        # 加载评分数据
        self.ratings_df = pd.read_csv(f"{self.data_dir}/processed_ratings.csv")
        
        # 加载商品元数据
        self.load_metadata()
        
        print(f"✅ 数据加载完成:")
        print(f"   - 评分数据: {len(self.ratings_df):,} 条")
        print(f"   - 商品元数据: {len(self.metadata_df):,} 个")
        
    def load_metadata(self):
        """加载并处理商品元数据"""
        metadata_file = f"{self.data_dir}/meta_Electronics.json"
        metadata_list = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = eval(line.strip())
                    
                    # 提取主要类别
                    main_category = ""
                    sub_category = ""
                    if item.get('categories'):
                        cats = item['categories'][0] if item['categories'] else []
                        if isinstance(cats, list) and len(cats) > 0:
                            main_category = cats[0] if len(cats) > 0 else ""
                            sub_category = cats[1] if len(cats) > 1 else ""
                    
                    metadata_list.append({
                        'item_id': item.get('asin', ''),
                        'title': item.get('title', ''),
                        'main_category': main_category,
                        'sub_category': sub_category,
                        'price': item.get('price', 0),
                        'brand': item.get('brand', ''),
                        'description': item.get('description', '')
                    })
                except:
                    continue
        
        self.metadata_df = pd.DataFrame(metadata_list)
        
        # 只保留有评分的商品
        rated_items = set(self.ratings_df['item_id'].unique())
        self.metadata_df = self.metadata_df[self.metadata_df['item_id'].isin(rated_items)]
        
        print(f"   - 有效商品元数据: {len(self.metadata_df):,} 个")
        
    def create_item_features(self):
        """创建商品特征矩阵"""
        print("🔧 创建商品特征矩阵...")
        
        # 处理缺失值
        self.metadata_df['price'] = self.metadata_df['price'].fillna(0)
        self.metadata_df['brand'] = self.metadata_df['brand'].fillna('Unknown')
        self.metadata_df['main_category'] = self.metadata_df['main_category'].fillna('Unknown')
        self.metadata_df['sub_category'] = self.metadata_df['sub_category'].fillna('Unknown')
        self.metadata_df['description'] = self.metadata_df['description'].fillna('')
        
        # 1. 类别特征 (One-hot编码)
        category_features = pd.get_dummies(self.metadata_df['main_category'], prefix='main_cat')
        sub_category_features = pd.get_dummies(self.metadata_df['sub_category'], prefix='sub_cat')
        
        # 2. 品牌特征 (One-hot编码，只保留出现频率较高的品牌)
        brand_counts = self.metadata_df['brand'].value_counts()
        top_brands = brand_counts[brand_counts >= 10].index  # 至少出现10次的品牌
        self.metadata_df['brand_grouped'] = self.metadata_df['brand'].apply(
            lambda x: x if x in top_brands else 'Other'
        )
        brand_features = pd.get_dummies(self.metadata_df['brand_grouped'], prefix='brand')
        
        # 3. 价格特征 (标准化)
        price_features = self.metadata_df[['price']].copy()
        price_features['price_normalized'] = self.scaler.fit_transform(price_features[['price']])
        
        # 4. 价格区间特征
        price_features['price_range'] = pd.cut(
            self.metadata_df['price'], 
            bins=[0, 20, 50, 100, 200, 500, float('inf')],
            labels=['0-20', '20-50', '50-100', '100-200', '200-500', '500+']
        )
        price_range_features = pd.get_dummies(price_features['price_range'], prefix='price_range')
        
        # 5. 文本特征 (TF-IDF)
        # 合并标题和描述
        text_content = (self.metadata_df['title'] + ' ' + self.metadata_df['description']).fillna('')
        
        # 使用TF-IDF向量化
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # 限制特征数量
            stop_words='english',
            ngram_range=(1, 2),  # 使用1-gram和2-gram
            min_df=2,  # 至少出现在2个文档中
            max_df=0.8  # 最多出现在80%的文档中
        )
        
        text_features = self.tfidf_vectorizer.fit_transform(text_content)
        text_features_df = pd.DataFrame(
            text_features.toarray(), 
            columns=[f'text_{i}' for i in range(text_features.shape[1])]
        )
        
        # 合并所有特征
        self.item_features = pd.concat([
            self.metadata_df[['item_id']].reset_index(drop=True),
            category_features.reset_index(drop=True),
            sub_category_features.reset_index(drop=True),
            brand_features.reset_index(drop=True),
            price_features[['price_normalized']].reset_index(drop=True),
            price_range_features.reset_index(drop=True),
            text_features_df.reset_index(drop=True)
        ], axis=1)
        
        print(f"   - 特征矩阵维度: {self.item_features.shape}")
        print(f"   - 类别特征: {len(category_features.columns)}")
        print(f"   - 子类别特征: {len(sub_category_features.columns)}")
        print(f"   - 品牌特征: {len(brand_features.columns)}")
        print(f"   - 文本特征: {text_features.shape[1]}")
        
    def compute_item_similarity(self):
        """计算商品相似度矩阵（优化内存使用）"""
        print("📊 准备特征矩阵（不预计算完整相似度矩阵以节省内存）...")

        # 提取特征矩阵（除了item_id列）
        feature_columns = [col for col in self.item_features.columns if col != 'item_id']
        self.feature_matrix = self.item_features[feature_columns].values

        # 标准化特征矩阵以便计算余弦相似度
        from sklearn.preprocessing import normalize
        self.feature_matrix = normalize(self.feature_matrix, norm='l2')

        print(f"   - 特征矩阵维度: {self.feature_matrix.shape}")
        print("   - 使用按需计算相似度的方式以节省内存")
        
    def get_user_profile(self, user_id):
        """构建用户偏好档案"""
        # 获取用户评分历史
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        # 只考虑高评分商品（4星及以上）
        high_rated_items = user_ratings[user_ratings['rating'] >= 4]['item_id'].tolist()
        
        # 获取这些商品的特征
        user_item_features = self.item_features[
            self.item_features['item_id'].isin(high_rated_items)
        ]
        
        if len(user_item_features) == 0:
            return None
        
        # 计算用户偏好向量（平均特征向量）
        feature_columns = [col for col in self.item_features.columns if col != 'item_id']
        user_profile = user_item_features[feature_columns].mean().values
        
        return user_profile, high_rated_items
    
    def get_content_recommendations(self, user_id, n_recommendations=10):
        """为用户生成基于内容的推荐"""
        print(f"🎯 为用户 {user_id} 生成基于内容的推荐...")
        
        # 构建用户偏好档案
        user_profile_result = self.get_user_profile(user_id)
        if user_profile_result is None:
            print("   - 用户没有足够的高评分历史，无法生成推荐")
            return []
        
        user_profile, user_items = user_profile_result
        
        # 获取用户已评分的所有商品
        user_rated_items = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'])
        
        # 计算用户偏好与所有商品的相似度
        feature_columns = [col for col in self.item_features.columns if col != 'item_id']
        item_features_matrix = self.item_features[feature_columns].values
        
        # 计算相似度
        similarities = cosine_similarity([user_profile], item_features_matrix)[0]
        
        # 创建推荐列表
        recommendations = []
        for idx, similarity in enumerate(similarities):
            item_id = self.item_features.iloc[idx]['item_id']
            
            # 排除用户已评分的商品
            if item_id not in user_rated_items:
                recommendations.append((item_id, similarity))
        
        # 按相似度排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # 添加商品信息
        final_recommendations = []
        for item_id, similarity in recommendations[:n_recommendations]:
            item_info = self.metadata_df[self.metadata_df['item_id'] == item_id]
            if not item_info.empty:
                item_info = item_info.iloc[0]
                final_recommendations.append({
                    'item_id': item_id,
                    'similarity_score': round(similarity, 4),
                    'title': item_info['title'],
                    'main_category': item_info['main_category'],
                    'sub_category': item_info['sub_category'],
                    'price': item_info['price'],
                    'brand': item_info['brand']
                })
        
        return final_recommendations
    
    def get_similar_items(self, item_id, n_similar=10):
        """获取与指定商品相似的商品（按需计算相似度）"""
        try:
            # 找到商品在特征矩阵中的索引
            item_idx = self.item_features[self.item_features['item_id'] == item_id].index[0]

            # 获取该商品的特征向量
            item_vector = self.feature_matrix[item_idx:item_idx+1]

            # 计算与所有商品的相似度
            similarities = np.dot(self.feature_matrix, item_vector.T).flatten()

            # 获取最相似的商品（排除自己）
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]

            similar_items = []
            for idx in similar_indices:
                similar_item_id = self.item_features.iloc[idx]['item_id']
                item_info = self.metadata_df[self.metadata_df['item_id'] == similar_item_id]
                if not item_info.empty:
                    item_info = item_info.iloc[0]
                    similar_items.append({
                        'item_id': similar_item_id,
                        'similarity_score': round(similarities[idx], 4),
                        'title': item_info['title'],
                        'main_category': item_info['main_category'],
                        'price': item_info['price'],
                        'brand': item_info['brand']
                    })

            return similar_items
        except:
            return []
    
    def save_model(self):
        """保存模型"""
        print("💾 保存基于内容的推荐模型...")
        
        # 保存特征矩阵
        self.item_features.to_csv(f"{self.data_dir}/item_features.csv", index=False)
        
        # 保存特征矩阵
        np.save(f"{self.data_dir}/feature_matrix.npy", self.feature_matrix)
        
        # 保存TF-IDF向量化器
        with open(f"{self.data_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # 保存标准化器
        with open(f"{self.data_dir}/price_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("✅ 模型保存完成!")
    
    def run_content_based_recommendation(self):
        """运行完整的基于内容的推荐流程"""
        print("🚀 开始基于内容的推荐系统训练...\n")
        
        self.load_data()
        self.create_item_features()
        self.compute_item_similarity()
        self.save_model()
        
        print(f"\n🎉 基于内容的推荐系统构建完成!")
        return self

# 注释掉测试代码，避免在导入时执行
# if __name__ == "__main__":
#     cb_recommender = ContentBasedRecommender()
#     model = cb_recommender.run_content_based_recommendation()
