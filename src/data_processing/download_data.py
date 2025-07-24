"""
Amazon Electronics 数据集下载脚本
"""
import os
import requests
import gzip
import json
from tqdm import tqdm
import pandas as pd

def download_file(url, filename):
    """下载文件并显示进度条"""
    print(f"正在下载: {filename}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    
    print(f"下载完成: {filename}")

def extract_gz_file(gz_filename, output_filename):
    """解压 .gz 文件"""
    print(f"正在解压: {gz_filename}")
    with gzip.open(gz_filename, 'rb') as f_in:
        with open(output_filename, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"解压完成: {output_filename}")

def download_amazon_electronics_data():
    """下载 Amazon Electronics 数据集"""
    
    # 创建数据目录
    data_dir = "../../data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 数据集URL（使用较小的样本数据集）
    urls = {
        'ratings': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv',
        'metadata': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz'
    }
    
    # 下载评分数据
    ratings_file = os.path.join(data_dir, 'ratings_Electronics.csv')
    if not os.path.exists(ratings_file):
        try:
            download_file(urls['ratings'], ratings_file)
        except Exception as e:
            print(f"下载评分数据失败: {e}")
            # 如果官方链接失败，创建示例数据
            create_sample_data()
            return
    
    # 下载商品元数据
    metadata_gz_file = os.path.join(data_dir, 'meta_Electronics.json.gz')
    metadata_file = os.path.join(data_dir, 'meta_Electronics.json')
    
    if not os.path.exists(metadata_file):
        try:
            download_file(urls['metadata'], metadata_gz_file)
            extract_gz_file(metadata_gz_file, metadata_file)
            os.remove(metadata_gz_file)  # 删除压缩文件
        except Exception as e:
            print(f"下载元数据失败: {e}")
            create_sample_data()
            return
    
    print("数据下载完成！")
    
    # 显示数据基本信息
    show_data_info()

def create_sample_data():
    """创建示例数据（如果无法下载真实数据）"""
    print("创建示例数据...")
    
    data_dir = "../../data"
    
    # 创建示例评分数据
    import numpy as np
    np.random.seed(42)
    
    n_users = 1000
    n_items = 500
    n_ratings = 10000
    
    user_ids = np.random.randint(1, n_users+1, n_ratings)
    item_ids = np.random.randint(1, n_items+1, n_ratings)
    ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.1, 0.2, 0.3, 0.3])
    timestamps = np.random.randint(1000000000, 1600000000, n_ratings)
    
    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    ratings_df.to_csv(os.path.join(data_dir, 'ratings_Electronics.csv'), index=False)
    
    # 创建示例商品元数据
    categories = ['Electronics', 'Computers', 'Cell Phones', 'Camera & Photo', 'TV & Video']
    brands = ['Samsung', 'Apple', 'Sony', 'LG', 'Canon', 'HP', 'Dell', 'Asus']
    
    metadata = []
    for i in range(1, n_items+1):
        metadata.append({
            'asin': f'B{i:07d}',
            'title': f'Product {i}',
            'category': np.random.choice(categories),
            'brand': np.random.choice(brands),
            'price': round(np.random.uniform(10, 1000), 2)
        })
    
    with open(os.path.join(data_dir, 'meta_Electronics.json'), 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    print("示例数据创建完成！")

def show_data_info():
    """显示数据基本信息"""
    data_dir = "../../data"

    # 评分数据信息
    ratings_file = os.path.join(data_dir, 'ratings_Electronics.csv')
    if os.path.exists(ratings_file):
        # Amazon数据集没有列名，需要手动指定
        ratings_df = pd.read_csv(ratings_file, names=['user_id', 'item_id', 'rating', 'timestamp'])
        print(f"\n评分数据信息:")
        print(f"- 总评分数: {len(ratings_df):,}")
        print(f"- 用户数: {ratings_df['user_id'].nunique():,}")
        print(f"- 商品数: {ratings_df['item_id'].nunique():,}")
        print(f"- 评分分布:\n{ratings_df['rating'].value_counts().sort_index()}")
    
    # 元数据信息
    metadata_file = os.path.join(data_dir, 'meta_Electronics.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata_count = sum(1 for line in f)
        print(f"\n商品元数据信息:")
        print(f"- 商品数量: {metadata_count:,}")

if __name__ == "__main__":
    download_amazon_electronics_data()
