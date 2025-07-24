"""
混合推荐系统
融合协同过滤和基于内容推荐的优势，提供更准确和多样化的推荐
"""
import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader
from collaborative_filtering import CollaborativeFilteringModel
from content_based_recommendation import ContentBasedRecommender
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class HybridRecommendationSystem:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.cf_model = None
        self.cb_model = None
        self.ratings_df = None
        self.metadata_df = None
        
        # 混合策略参数
        self.cf_weight = 0.6  # 协同过滤权重
        self.cb_weight = 0.4  # 基于内容推荐权重
        
    def load_models(self):
        """加载协同过滤和基于内容推荐模型"""
        print("📂 加载混合推荐系统组件...")
        
        # 加载协同过滤模型
        print("   - 加载协同过滤模型...")
        self.cf_model = CollaborativeFilteringModel(self.data_dir)
        ratings_df = self.cf_model.load_processed_data()

        # 重新创建trainset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_df[['user_id', 'item_id', 'rating']],
            reader
        )
        self.cf_model.trainset = data.build_full_trainset()
        
        # 加载训练好的SVD模型
        with open(f"{self.data_dir}/svd_model.pkl", 'rb') as f:
            self.cf_model.model = pickle.load(f)
        
        # 加载基于内容推荐模型
        print("   - 加载基于内容推荐模型...")
        self.cb_model = ContentBasedRecommender(self.data_dir)
        self.cb_model.load_data()
        
        # 加载保存的特征
        self.cb_model.item_features = pd.read_csv(f"{self.data_dir}/item_features.csv")
        self.cb_model.feature_matrix = np.load(f"{self.data_dir}/feature_matrix.npy")
        
        with open(f"{self.data_dir}/tfidf_vectorizer.pkl", 'rb') as f:
            self.cb_model.tfidf_vectorizer = pickle.load(f)
        
        with open(f"{self.data_dir}/price_scaler.pkl", 'rb') as f:
            self.cb_model.scaler = pickle.load(f)
        
        # 共享数据
        self.ratings_df = ratings_df
        self.metadata_df = self.cb_model.metadata_df
        
        print("✅ 混合推荐系统组件加载完成!")
        
    def get_cf_recommendations(self, user_id, n_recommendations=20):
        """获取协同过滤推荐"""
        try:
            # 获取用户已评分的商品
            user_items = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'])
            
            # 获取所有商品
            all_items = set(self.ratings_df['item_id'].unique())
            
            # 找到用户未评分的商品
            unrated_items = all_items - user_items
            
            # 预测评分
            predictions = []
            for item_id in unrated_items:
                pred = self.cf_model.model.predict(user_id, item_id)
                predictions.append((item_id, pred.est))
            
            # 按预测评分排序
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predictions[:n_recommendations]
        except:
            return []
    
    def get_cb_recommendations(self, user_id, n_recommendations=20):
        """获取基于内容的推荐"""
        try:
            content_recs = self.cb_model.get_content_recommendations(user_id, n_recommendations)
            return [(rec['item_id'], rec['similarity_score']) for rec in content_recs]
        except:
            return []
    
    def normalize_scores(self, scores, method='min_max'):
        """标准化分数到[0,1]区间"""
        if not scores:
            return scores
        
        values = [score for _, score in scores]
        
        if method == 'min_max':
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [(item, 0.5) for item, _ in scores]
            return [(item, (score - min_val) / (max_val - min_val)) for item, score in scores]
        
        elif method == 'z_score':
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return [(item, 0.5) for item, _ in scores]
            normalized = [(item, (score - mean_val) / std_val) for item, score in scores]
            # 转换到[0,1]区间
            norm_values = [score for _, score in normalized]
            min_norm, max_norm = min(norm_values), max(norm_values)
            if max_norm == min_norm:
                return [(item, 0.5) for item, _ in normalized]
            return [(item, (score - min_norm) / (max_norm - min_norm)) for item, score in normalized]
    
    def hybrid_weighted_combination(self, user_id, n_recommendations=10):
        """加权组合混合推荐策略"""
        print(f"🔄 为用户 {user_id} 生成混合推荐 (加权组合)...")
        
        # 获取两种推荐
        cf_recs = self.get_cf_recommendations(user_id, 50)
        cb_recs = self.get_cb_recommendations(user_id, 50)
        
        if not cf_recs and not cb_recs:
            return []
        
        # 标准化分数
        cf_normalized = self.normalize_scores(cf_recs)
        cb_normalized = self.normalize_scores(cb_recs)
        
        # 创建商品分数字典
        item_scores = defaultdict(float)
        item_sources = defaultdict(list)
        
        # 添加协同过滤分数
        for item_id, score in cf_normalized:
            item_scores[item_id] += self.cf_weight * score
            item_sources[item_id].append('CF')
        
        # 添加基于内容推荐分数
        for item_id, score in cb_normalized:
            item_scores[item_id] += self.cb_weight * score
            item_sources[item_id].append('CB')
        
        # 排序并获取top-N
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 添加商品信息
        recommendations = []
        for item_id, hybrid_score in sorted_items[:n_recommendations]:
            item_info = self.metadata_df[self.metadata_df['item_id'] == item_id]
            if not item_info.empty:
                item_info = item_info.iloc[0]
                recommendations.append({
                    'item_id': item_id,
                    'hybrid_score': round(hybrid_score, 4),
                    'sources': item_sources[item_id],
                    'title': item_info['title'],
                    'main_category': item_info['main_category'],
                    'sub_category': item_info['sub_category'],
                    'price': item_info['price'],
                    'brand': item_info['brand']
                })
        
        return recommendations
    
    def hybrid_switching_strategy(self, user_id, n_recommendations=10):
        """切换混合推荐策略"""
        print(f"🔀 为用户 {user_id} 生成混合推荐 (切换策略)...")
        
        # 分析用户特征决定使用哪种策略
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        # 用户活跃度
        user_activity = len(user_ratings)
        
        # 评分多样性（评分的标准差）
        rating_diversity = user_ratings['rating'].std() if len(user_ratings) > 1 else 0
        
        # 决策逻辑
        if user_activity >= 20 and rating_diversity > 1.0:
            # 活跃且评分多样的用户：主要使用协同过滤
            print(f"   - 用户类型: 活跃多样用户，主要使用协同过滤")
            cf_recs = self.get_cf_recommendations(user_id, n_recommendations)
            return self._format_recommendations(cf_recs, 'CF')
        
        elif user_activity < 10:
            # 新用户或不活跃用户：主要使用基于内容推荐
            print(f"   - 用户类型: 新用户/不活跃用户，主要使用基于内容推荐")
            cb_recs = self.get_cb_recommendations(user_id, n_recommendations)
            return self._format_recommendations(cb_recs, 'CB')
        
        else:
            # 中等活跃用户：使用加权组合
            print(f"   - 用户类型: 中等活跃用户，使用加权组合")
            return self.hybrid_weighted_combination(user_id, n_recommendations)
    
    def hybrid_mixed_strategy(self, user_id, n_recommendations=10):
        """混合策略：分别从两种方法中选择部分推荐"""
        print(f"🎯 为用户 {user_id} 生成混合推荐 (混合策略)...")
        
        # 分配比例
        cf_count = int(n_recommendations * 0.6)
        cb_count = n_recommendations - cf_count
        
        # 获取推荐
        cf_recs = self.get_cf_recommendations(user_id, cf_count * 2)
        cb_recs = self.get_cb_recommendations(user_id, cb_count * 2)
        
        # 格式化推荐
        cf_formatted = self._format_recommendations(cf_recs[:cf_count], 'CF')
        cb_formatted = self._format_recommendations(cb_recs[:cb_count], 'CB')
        
        # 合并并打乱顺序
        mixed_recs = cf_formatted + cb_formatted
        np.random.shuffle(mixed_recs)
        
        return mixed_recs
    
    def _format_recommendations(self, recs, source):
        """格式化推荐结果"""
        recommendations = []
        for item_id, score in recs:
            item_info = self.metadata_df[self.metadata_df['item_id'] == item_id]
            if not item_info.empty:
                item_info = item_info.iloc[0]
                recommendations.append({
                    'item_id': item_id,
                    'score': round(score, 4),
                    'source': source,
                    'title': item_info['title'],
                    'main_category': item_info['main_category'],
                    'sub_category': item_info['sub_category'],
                    'price': item_info['price'],
                    'brand': item_info['brand']
                })
        return recommendations
    
    def analyze_user_for_strategy(self, user_id):
        """分析用户特征以选择最佳策略"""
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        analysis = {
            'user_id': user_id,
            'total_ratings': len(user_ratings),
            'avg_rating': user_ratings['rating'].mean(),
            'rating_std': user_ratings['rating'].std(),
            'high_ratings_count': len(user_ratings[user_ratings['rating'] >= 4]),
            'categories_count': 0
        }
        
        # 分析用户购买的商品类别多样性
        user_items = user_ratings['item_id'].tolist()
        user_metadata = self.metadata_df[self.metadata_df['item_id'].isin(user_items)]
        categories = set()
        for _, row in user_metadata.iterrows():
            if row['main_category']:
                categories.add(row['main_category'])
        analysis['categories_count'] = len(categories)
        
        return analysis
    
    def get_hybrid_recommendations(self, user_id, strategy='weighted', n_recommendations=10):
        """获取混合推荐（主接口）"""
        if strategy == 'weighted':
            return self.hybrid_weighted_combination(user_id, n_recommendations)
        elif strategy == 'switching':
            return self.hybrid_switching_strategy(user_id, n_recommendations)
        elif strategy == 'mixed':
            return self.hybrid_mixed_strategy(user_id, n_recommendations)
        else:
            raise ValueError("Strategy must be 'weighted', 'switching', or 'mixed'")
    
    def set_weights(self, cf_weight, cb_weight):
        """设置协同过滤和基于内容推荐的权重"""
        total = cf_weight + cb_weight
        self.cf_weight = cf_weight / total
        self.cb_weight = cb_weight / total
        print(f"权重设置: CF={self.cf_weight:.2f}, CB={self.cb_weight:.2f}")

# 注释掉测试代码，避免在导入时执行
# if __name__ == "__main__":
#     # 初始化混合推荐系统
#     hybrid_system = HybridRecommendationSystem()
#     hybrid_system.load_models()
#
#     print("🎉 混合推荐系统构建完成!")
