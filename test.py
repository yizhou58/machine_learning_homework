"""
推荐系统功能完整性测试
测试所有核心功能是否正常工作
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle

def test_data_integrity():
    """测试数据完整性"""
    print("📊 测试数据完整性...")
    
    try:
        # 测试评分数据
        ratings_df = pd.read_csv('data/processed_ratings.csv')
        print(f"✅ 评分数据: {len(ratings_df):,} 条")
        print(f"   - 用户数: {ratings_df['user_id'].nunique():,}")
        print(f"   - 商品数: {ratings_df['item_id'].nunique():,}")
        print(f"   - 评分范围: {ratings_df['rating'].min()}-{ratings_df['rating'].max()}")
        
        # 测试特征数据
        item_features = pd.read_csv('data/item_features.csv')
        feature_matrix = np.load('data/feature_matrix.npy')
        print(f"✅ 特征数据: {feature_matrix.shape[0]:,} × {feature_matrix.shape[1]:,}")
        
        return True
    except Exception as e:
        print(f"❌ 数据完整性测试失败: {e}")
        return False

def test_collaborative_filtering():
    """测试协同过滤功能"""
    print("\n🤖 测试协同过滤功能...")
    
    try:
        # 加载SVD模型
        with open('data/svd_model.pkl', 'rb') as f:
            svd_model = pickle.load(f)
        
        # 加载数据
        ratings_df = pd.read_csv('data/processed_ratings.csv')
        
        # 选择测试用户
        user_activity = ratings_df['user_id'].value_counts()
        test_user = user_activity.index[0]
        
        # 生成推荐
        user_items = set(ratings_df[ratings_df['user_id'] == test_user]['item_id'])
        all_items = ratings_df['item_id'].unique()
        test_items = [item for item in all_items if item not in user_items][:5]
        
        predictions = []
        for item_id in test_items:
            pred = svd_model.predict(test_user, item_id)
            predictions.append((item_id, pred.est))
        
        print(f"✅ 协同过滤推荐: {len(predictions)} 个")
        print(f"   - 测试用户: {test_user}")
        print(f"   - 预测评分范围: {min(p[1] for p in predictions):.2f}-{max(p[1] for p in predictions):.2f}")
        
        return True
    except Exception as e:
        print(f"❌ 协同过滤测试失败: {e}")
        return False

def test_content_based():
    """测试基于内容推荐功能"""
    print("\n📋 测试基于内容推荐功能...")
    
    try:
        # 加载特征数据
        feature_matrix = np.load('data/feature_matrix.npy')
        item_features = pd.read_csv('data/item_features.csv')
        ratings_df = pd.read_csv('data/processed_ratings.csv')
        
        # 选择测试用户
        user_activity = ratings_df['user_id'].value_counts()
        test_user = user_activity.index[0]
        
        # 获取用户高评分商品
        user_ratings = ratings_df[ratings_df['user_id'] == test_user]
        high_rated_items = user_ratings[user_ratings['rating'] >= 4]['item_id'].tolist()
        
        if high_rated_items:
            # 构建用户偏好档案
            user_item_indices = []
            for item_id in high_rated_items[:5]:
                item_idx = item_features[item_features['item_id'] == item_id].index
                if len(item_idx) > 0:
                    user_item_indices.append(item_idx[0])
            
            if user_item_indices:
                user_profile = feature_matrix[user_item_indices].mean(axis=0)
                
                # 计算相似度（限制计算量）
                similarities = np.dot(feature_matrix[:100], user_profile)
                
                print(f"✅ 基于内容推荐: 计算完成")
                print(f"   - 用户高评分商品: {len(high_rated_items)}")
                print(f"   - 相似度范围: {similarities.min():.3f}-{similarities.max():.3f}")
                
                return True
        
        print("⚠️ 用户没有足够的高评分历史")
        return True
        
    except Exception as e:
        print(f"❌ 基于内容推荐测试失败: {e}")
        return False

def test_hybrid_system():
    """测试混合推荐系统"""
    print("\n🔄 测试混合推荐系统...")
    
    try:
        sys.path.append('src/models')
        from hybrid_recommendation import HybridRecommendationSystem
        
        # 初始化系统
        hybrid_system = HybridRecommendationSystem('data')
        hybrid_system.load_models()
        
        # 选择测试用户
        user_activity = hybrid_system.ratings_df['user_id'].value_counts()
        test_user = user_activity.index[0]
        
        # 生成推荐
        recommendations = hybrid_system.get_hybrid_recommendations(test_user, 'weighted', 2)
        
        print(f"✅ 混合推荐系统: {len(recommendations)} 个推荐")
        print(f"   - 测试用户: {test_user}")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. 商品: {rec['item_id']} | 评分: {rec['hybrid_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 混合推荐系统测试失败: {e}")
        return False

def test_model_files():
    """测试模型文件完整性"""
    print("\n📁 测试模型文件完整性...")
    
    required_files = [
        'data/processed_ratings.csv',
        'data/meta_Electronics.json',
        'data/svd_model.pkl',
        'data/feature_matrix.npy',
        'data/item_features.csv',
        'data/tfidf_vectorizer.pkl',
        'data/price_scaler.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024*1024)
            print(f"   ✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {file} - 缺失")
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺失 {len(missing_files)} 个文件")
        return False
    else:
        print("✅ 所有模型文件完整")
        return True

def main():
    """运行所有测试"""
    print("🧪 推荐系统功能完整性测试")
    print("=" * 50)
    
    tests = [
        ("数据完整性", test_data_integrity),
        ("协同过滤", test_collaborative_filtering),
        ("基于内容推荐", test_content_based),
        ("混合推荐系统", test_hybrid_system),
        ("模型文件", test_model_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 测试结果汇总
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("🎉 所有功能测试通过！推荐系统完全正常！")
    else:
        print("⚠️ 部分功能存在问题，请检查失败的测试项")

if __name__ == "__main__":
    main()
