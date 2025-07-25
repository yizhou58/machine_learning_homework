"""
Neural Collaborative Filtering (NCF) 深度学习推荐模型
创新点：融合多模态特征的神经协同过滤

主要特性：
1. 神经矩阵分解 (Neural Matrix Factorization)
2. 多层感知机 (Multi-Layer Perceptron) 
3. 多模态特征融合 (Multi-modal Feature Fusion)
4. 注意力机制 (Attention Mechanism)
5. 对比学习 (Contrastive Learning)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MultiModalDataset(Dataset):
    """多模态推荐数据集"""
    
    def __init__(self, ratings_df, item_features, user_features=None):
        self.ratings = ratings_df
        self.item_features = item_features
        self.user_features = user_features
        
        # 创建用户和商品的ID映射
        self.user_ids = sorted(ratings_df['user_id'].unique())
        self.item_ids = sorted(ratings_df['item_id'].unique())
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        # 准备训练数据
        self.prepare_data()
        
    def prepare_data(self):
        """准备训练数据"""
        self.user_indices = []
        self.item_indices = []
        self.ratings_list = []
        self.item_features_list = []
        
        for _, row in self.ratings.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['item_id']]
            
            # 获取商品特征
            item_feature = self.item_features[
                self.item_features['item_id'] == row['item_id']
            ]
            
            if not item_feature.empty:
                # 提取特征向量（除了item_id列）
                feature_cols = [col for col in item_feature.columns if col != 'item_id']
                features = item_feature[feature_cols].values[0]
                
                self.user_indices.append(user_idx)
                self.item_indices.append(item_idx)
                self.ratings_list.append(row['rating'])
                self.item_features_list.append(features)
        
        # 转换为numpy数组
        self.user_indices = np.array(self.user_indices)
        self.item_indices = np.array(self.item_indices)
        self.ratings_list = np.array(self.ratings_list, dtype=np.float32)
        self.item_features_list = np.array(self.item_features_list, dtype=np.float32)
        
    def __len__(self):
        return len(self.user_indices)
    
    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_indices[idx], dtype=torch.long),
            'item_id': torch.tensor(self.item_indices[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings_list[idx], dtype=torch.float32),
            'item_features': torch.tensor(self.item_features_list[idx], dtype=torch.float32)
        }

class AttentionLayer(nn.Module):
    """注意力机制层"""
    
    def __init__(self, input_dim, attention_dim=64):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        weighted_output = torch.sum(x * attention_weights, dim=1)  # (batch_size, input_dim)
        return weighted_output, attention_weights

class MultiModalNCF(nn.Module):
    """多模态神经协同过滤模型"""
    
    def __init__(self, num_users, num_items, item_feature_dim, 
                 embedding_dim=64, hidden_dims=[128, 64, 32], dropout=0.2):
        super(MultiModalNCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # 用户和商品嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 商品特征处理网络
        self.item_feature_net = nn.Sequential(
            nn.Linear(item_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 注意力机制 - 融合不同模态特征
        self.attention = AttentionLayer(embedding_dim)
        
        # GMF (Generalized Matrix Factorization) 分支
        self.gmf_layer = nn.Linear(embedding_dim, 1)
        
        # MLP (Multi-Layer Perceptron) 分支
        mlp_layers = []
        input_dim = embedding_dim * 3  # user + item + item_features
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp_output = nn.Linear(hidden_dims[-1], 1)
        
        # 最终融合层
        self.final_layer = nn.Sequential(
            nn.Linear(2, 32),  # GMF + MLP outputs
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, user_ids, item_ids, item_features):
        # 获取嵌入向量
        user_emb = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        item_emb = self.item_embedding(item_ids)  # (batch_size, embedding_dim)
        
        # 处理商品特征
        item_feat_emb = self.item_feature_net(item_features)  # (batch_size, embedding_dim)
        
        # 注意力机制融合多模态特征
        # 将三种特征堆叠
        multi_modal_features = torch.stack([user_emb, item_emb, item_feat_emb], dim=1)
        attended_features, attention_weights = self.attention(multi_modal_features)
        
        # GMF分支：元素级乘积
        gmf_vector = user_emb * item_emb
        gmf_output = self.gmf_layer(gmf_vector)
        
        # MLP分支：特征拼接
        mlp_vector = torch.cat([user_emb, item_emb, item_feat_emb], dim=1)
        mlp_hidden = self.mlp(mlp_vector)
        mlp_output = self.mlp_output(mlp_hidden)
        
        # 融合GMF和MLP输出
        final_input = torch.cat([gmf_output, mlp_output], dim=1)
        rating_pred = self.final_layer(final_input) * 4 + 1  # 缩放到1-5范围
        
        return {
            'rating': rating_pred.squeeze(),
            'user_emb': user_emb,
            'item_emb': item_emb,
            'item_feat_emb': item_feat_emb,
            'attention_weights': attention_weights,
            'user_projection': self.projection_head(user_emb),
            'item_projection': self.projection_head(item_emb)
        }

class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, user_proj, item_proj, ratings):
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(user_proj, item_proj.T) / self.temperature
        
        # 创建正样本标签（高评分为正样本）
        positive_mask = (ratings.unsqueeze(1) >= 4.0) & (ratings.unsqueeze(0) >= 4.0)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        positive_sim = exp_sim * positive_mask.float()
        
        loss = -torch.log(positive_sim.sum(dim=1) / exp_sim.sum(dim=1) + 1e-8)
        return loss.mean()

class NeuralCollaborativeFiltering:
    """神经协同过滤推荐系统"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 使用设备: {self.device}")
        
        # 损失函数
        self.rating_criterion = nn.MSELoss()
        self.contrastive_criterion = ContrastiveLoss()
        
    def load_data(self):
        """加载数据"""
        print("📂 加载数据...")
        
        # 加载评分数据
        self.ratings_df = pd.read_csv(f'{self.data_dir}/processed_ratings.csv')
        
        # 加载商品特征
        self.item_features = pd.read_csv(f'{self.data_dir}/item_features.csv')
        
        print(f"   - 评分数据: {len(self.ratings_df):,} 条")
        print(f"   - 商品特征: {self.item_features.shape}")
        
        # 数据采样（用于快速训练）
        if len(self.ratings_df) > 100000:
            print("   - 采样数据以加速训练...")
            self.ratings_df = self.ratings_df.sample(n=100000, random_state=42)
        
    def prepare_datasets(self, test_size=0.2):
        """准备训练和测试数据集"""
        print("🔧 准备数据集...")
        
        # 划分训练和测试集
        train_ratings, test_ratings = train_test_split(
            self.ratings_df, test_size=test_size, random_state=42
        )
        
        # 创建数据集
        self.train_dataset = MultiModalDataset(train_ratings, self.item_features)
        self.test_dataset = MultiModalDataset(test_ratings, self.item_features)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=512, shuffle=True, num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=512, shuffle=False, num_workers=0
        )
        
        print(f"   - 训练集: {len(self.train_dataset):,} 条")
        print(f"   - 测试集: {len(self.test_dataset):,} 条")
        
    def build_model(self):
        """构建模型"""
        print("🏗️ 构建神经网络模型...")
        
        num_users = len(self.train_dataset.user_ids)
        num_items = len(self.train_dataset.item_ids)
        item_feature_dim = self.item_features.shape[1] - 1  # 除去item_id列
        
        self.model = MultiModalNCF(
            num_users=num_users,
            num_items=num_items,
            item_feature_dim=item_feature_dim,
            embedding_dim=64,
            hidden_dims=[128, 64, 32],
            dropout=0.3
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
        
        print(f"   - 用户数: {num_users:,}")
        print(f"   - 商品数: {num_items:,}")
        print(f"   - 特征维度: {item_feature_dim}")
        print(f"   - 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_model(self, epochs=20):
        """训练模型"""
        print("🚀 开始训练神经网络...")
        
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_rating_loss = 0.0
            train_contrastive_loss = 0.0
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(user_ids, item_ids, item_features)
                
                # 计算损失
                rating_loss = self.rating_criterion(outputs['rating'], ratings)
                contrastive_loss = self.contrastive_criterion(
                    outputs['user_projection'], 
                    outputs['item_projection'], 
                    ratings
                )
                
                # 总损失 = 评分损失 + 对比学习损失
                total_loss = rating_loss + 0.1 * contrastive_loss
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += total_loss.item()
                train_rating_loss += rating_loss.item()
                train_contrastive_loss += contrastive_loss.item()
            
            # 验证阶段
            test_loss = self.evaluate_model()
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录损失
            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            test_losses.append(test_loss)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练损失: {avg_train_loss:.4f}")
            print(f"  - 评分损失: {train_rating_loss/len(self.train_loader):.4f}")
            print(f"  - 对比损失: {train_contrastive_loss/len(self.train_loader):.4f}")
            print(f"  测试损失: {test_loss:.4f}")
            print(f"  学习率: {self.scheduler.get_last_lr()[0]:.6f}")
            print()
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, test_losses)
        
    def evaluate_model(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.test_loader:
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                
                outputs = self.model(user_ids, item_ids, item_features)
                loss = self.rating_criterion(outputs['rating'], ratings)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def plot_training_curves(self, train_losses, test_losses):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', color='blue')
        plt.plot(test_losses, label='测试损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('神经协同过滤训练曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.data_dir}/ncf_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_recommendations(self, user_id, n_recommendations=10):
        """为用户生成推荐"""
        self.model.eval()
        
        if user_id not in self.train_dataset.user_to_idx:
            print(f"用户 {user_id} 不在训练集中")
            return []
        
        user_idx = self.train_dataset.user_to_idx[user_id]
        
        # 获取用户已评分的商品
        user_rated_items = set(
            self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id']
        )
        
        # 候选商品（未评分的商品）
        candidate_items = [
            item_id for item_id in self.train_dataset.item_ids 
            if item_id not in user_rated_items
        ]
        
        if not candidate_items:
            return []
        
        recommendations = []
        
        with torch.no_grad():
            for item_id in candidate_items[:1000]:  # 限制候选数量
                if item_id not in self.train_dataset.item_to_idx:
                    continue
                    
                item_idx = self.train_dataset.item_to_idx[item_id]
                
                # 获取商品特征
                item_feature = self.item_features[
                    self.item_features['item_id'] == item_id
                ]
                
                if item_feature.empty:
                    continue
                
                feature_cols = [col for col in item_feature.columns if col != 'item_id']
                features = torch.tensor(
                    item_feature[feature_cols].values[0], 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
                
                # 预测评分
                outputs = self.model(user_tensor, item_tensor, features)
                predicted_rating = outputs['rating'].item()
                
                recommendations.append({
                    'item_id': item_id,
                    'predicted_rating': predicted_rating
                })
        
        # 按预测评分排序
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def save_model(self):
        """保存模型"""
        print("💾 保存神经协同过滤模型...")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_to_idx': self.train_dataset.user_to_idx,
            'item_to_idx': self.train_dataset.item_to_idx,
            'model_config': {
                'num_users': len(self.train_dataset.user_ids),
                'num_items': len(self.train_dataset.item_ids),
                'item_feature_dim': self.item_features.shape[1] - 1
            }
        }, f'{self.data_dir}/ncf_model.pth')
        
        print("   ✅ 模型保存完成")

def main():
    """主函数"""
    print("🧠 神经协同过滤推荐系统")
    print("=" * 50)

    # 确定数据目录路径
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'data')

    # 初始化系统
    ncf = NeuralCollaborativeFiltering(data_dir)
    
    # 加载数据
    ncf.load_data()
    
    # 准备数据集
    ncf.prepare_datasets()
    
    # 构建模型
    ncf.build_model()
    
    # 训练模型
    ncf.train_model(epochs=15)
    
    # 保存模型
    ncf.save_model()
    
    # 测试推荐
    print("🎯 测试推荐生成...")
    user_activity = ncf.ratings_df['user_id'].value_counts()
    test_user = user_activity.index[0]
    
    recommendations = ncf.get_recommendations(test_user, 5)
    
    print(f"\n为用户 {test_user} 的推荐:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. 商品: {rec['item_id']} | 预测评分: {rec['predicted_rating']:.3f}")
    
    print("\n🎉 神经协同过滤训练完成！")

if __name__ == "__main__":
    main()
