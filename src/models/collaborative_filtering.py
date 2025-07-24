"""
协同过滤推荐模型
实现基于矩阵分解（SVD）的协同过滤算法
"""
import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFilteringModel:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.model = None
        self.trainset = None
        self.testset = None
        self.predictions = None
        self.data = None
        
    def load_processed_data(self):
        """加载预处理后的数据"""
        print("📂 加载预处理数据...")
        
        # 加载评分数据
        ratings_df = pd.read_csv(f"{self.data_dir}/processed_ratings.csv")
        
        # 为Surprise库准备数据格式
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(
            ratings_df[['user_id', 'item_id', 'rating']], 
            reader
        )
        
        print(f"✅ 数据加载完成: {len(ratings_df):,} 条评分")
        return ratings_df
    
    def split_data(self, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        print(f"📊 划分数据集 (测试集比例: {test_size})...")
        
        self.trainset, self.testset = train_test_split(
            self.data, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"   - 训练集大小: {self.trainset.n_ratings:,}")
        print(f"   - 测试集大小: {len(self.testset):,}")
    
    def train_svd_model(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """训练SVD模型"""
        print("🧠 训练SVD协同过滤模型...")
        print(f"   - 潜在因子数: {n_factors}")
        print(f"   - 训练轮数: {n_epochs}")
        print(f"   - 学习率: {lr_all}")
        print(f"   - 正则化参数: {reg_all}")
        
        # 初始化SVD模型
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42,
            verbose=True
        )
        
        # 训练模型
        self.model.fit(self.trainset)
        print("✅ 模型训练完成!")
    
    def evaluate_model(self):
        """评估模型性能"""
        print("\n📈 评估模型性能...")
        
        # 在测试集上进行预测
        self.predictions = self.model.test(self.testset)
        
        # 计算评估指标
        rmse = accuracy.rmse(self.predictions, verbose=False)
        mae = accuracy.mae(self.predictions, verbose=False)
        
        print(f"   - RMSE: {rmse:.4f}")
        print(f"   - MAE: {mae:.4f}")
        
        return rmse, mae
    
    def cross_validation(self, cv=5):
        """交叉验证"""
        print(f"\n🔄 进行 {cv} 折交叉验证...")
        
        cv_results = cross_validate(
            self.model, 
            self.data, 
            measures=['RMSE', 'MAE'], 
            cv=cv, 
            verbose=True
        )
        
        print(f"   - 平均 RMSE: {cv_results['test_rmse'].mean():.4f} (±{cv_results['test_rmse'].std():.4f})")
        print(f"   - 平均 MAE: {cv_results['test_mae'].mean():.4f} (±{cv_results['test_mae'].std():.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self):
        """超参数调优"""
        print("\n🔧 进行超参数调优...")
        
        param_grid = {
            'n_factors': [50, 100, 150],
            'n_epochs': [10, 20, 30],
            'lr_all': [0.002, 0.005, 0.01],
            'reg_all': [0.01, 0.02, 0.05]
        }
        
        gs = GridSearchCV(
            SVD, 
            param_grid, 
            measures=['rmse', 'mae'], 
            cv=3,
            n_jobs=-1
        )
        
        gs.fit(self.data)
        
        print(f"   - 最佳 RMSE: {gs.best_score['rmse']:.4f}")
        print(f"   - 最佳参数: {gs.best_params['rmse']}")
        
        # 使用最佳参数重新训练模型
        self.model = gs.best_estimator['rmse']
        self.model.fit(self.trainset)
        
        return gs.best_params['rmse']
    
    def get_user_recommendations(self, user_id, n_recommendations=10):
        """为指定用户生成推荐"""
        # 获取用户已评分的商品
        user_items = set()
        for (uid, iid, rating) in self.trainset.all_ratings():
            if self.trainset.to_raw_uid(uid) == user_id:
                user_items.add(self.trainset.to_raw_iid(iid))
        
        # 获取所有商品
        all_items = set()
        for (uid, iid, rating) in self.trainset.all_ratings():
            all_items.add(self.trainset.to_raw_iid(iid))
        
        # 找到用户未评分的商品
        unrated_items = all_items - user_items
        
        # 预测评分
        predictions = []
        for item_id in unrated_items:
            pred = self.model.predict(user_id, item_id)
            predictions.append((item_id, pred.est))
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def analyze_predictions(self):
        """分析预测结果"""
        print("\n📊 分析预测结果...")
        
        # 提取真实评分和预测评分
        true_ratings = [pred.r_ui for pred in self.predictions]
        pred_ratings = [pred.est for pred in self.predictions]
        
        # 计算预测误差分布
        errors = [abs(true - pred) for true, pred in zip(true_ratings, pred_ratings)]
        
        print(f"   - 预测评分范围: {min(pred_ratings):.2f} - {max(pred_ratings):.2f}")
        print(f"   - 平均绝对误差: {np.mean(errors):.4f}")
        print(f"   - 误差标准差: {np.std(errors):.4f}")
        
        # 按真实评分分组分析
        rating_errors = defaultdict(list)
        for pred in self.predictions:
            rating_errors[pred.r_ui].append(abs(pred.r_ui - pred.est))
        
        print(f"   - 各评分等级的平均误差:")
        for rating in sorted(rating_errors.keys()):
            avg_error = np.mean(rating_errors[rating])
            print(f"     {rating}星: {avg_error:.4f}")
    
    def save_model(self):
        """保存训练好的模型"""
        print("\n💾 保存模型...")
        
        with open(f"{self.data_dir}/svd_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        print("✅ 模型保存完成!")
    
    def run_collaborative_filtering(self, tune_hyperparams=False):
        """运行完整的协同过滤流程"""
        print("🚀 开始协同过滤推荐系统训练...\n")
        
        # 加载数据
        ratings_df = self.load_processed_data()
        
        # 划分数据集
        self.split_data()
        
        # 训练模型
        if tune_hyperparams:
            best_params = self.hyperparameter_tuning()
        else:
            self.train_svd_model()
        
        # 评估模型
        rmse, mae = self.evaluate_model()
        
        # 交叉验证
        cv_results = self.cross_validation()
        
        # 分析预测结果
        self.analyze_predictions()
        
        # 保存模型
        self.save_model()
        
        print(f"\n🎉 协同过滤模型训练完成!")
        print(f"   - 最终 RMSE: {rmse:.4f}")
        print(f"   - 最终 MAE: {mae:.4f}")
        
        return self.model

# 注释掉测试代码，避免在导入时执行
# if __name__ == "__main__":
#     cf_model = CollaborativeFilteringModel()
#     model = cf_model.run_collaborative_filtering(tune_hyperparams=False)
