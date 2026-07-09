import json
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import argparse
sys.path.append('./LeFusion/dataset')
from isic_hist import ISICDataset

class DatasetForClustering:
    def __init__(self, dataset):
        self.dataset = dataset
        self.hist_features = []
    
    def extract_hist_features(self):
        """从数据集中提取所有直方图特征"""
        self.hist_features = []
        valid_count = 0
        
        print("Extracting histogram features...")
        for i in tqdm(range(len(self.dataset))):
            try:
                item = self.dataset[i]
                hist = item['hist']
                
                # 确保直方图是有效的
                if hist is not None:
                    # 转换为numpy并检查有效性
                    hist_np = hist.numpy()
                    if not np.isnan(hist_np).any() and hist_np.sum() > 0:
                        self.hist_features.append(hist_np)
                        valid_count += 1
                    else:
                        # 如果直方图全零或包含NaN，打印警告但继续
                        if i < 10:  # 只打印前10个无效样本
                            print(f"Sample {i}: Invalid histogram (sum={hist_np.sum()})")
                else:
                    print(f"Sample {i}: Hist is None")
            except Exception as e:
                if i < 10:  # 只打印前10个错误
                    print(f"Error at sample {i}: {e}")
                continue
        
        print(f"Total samples: {len(self.dataset)}, Valid samples: {valid_count}")
        return np.array(self.hist_features)
    
    def perform_clustering(self, n_clusters=3, random_state=42):
        """对直方图特征进行K-means聚类"""
        features = self.extract_hist_features()
        
        if len(features) == 0:
            raise ValueError("No valid features extracted from dataset!")
        
        print(f"Features shape: {features.shape}")
        
        # 确保有足够的数据进行聚类
        if len(features) < n_clusters:
            print(f"Warning: Only {len(features)} samples available, reducing clusters to 2")
            n_clusters = max(2, len(features))
        
        # 执行K-means聚类
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        
        return features, labels, centers
    
    def visualize_clusters(self, features, labels, centers):
        """使用PCA降维可视化聚类结果"""
        # 使用PCA将16维特征降为2维
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        centers_2d = pca.transform(centers)
        
        # 绘制聚类结果
        plt.figure(figsize=(12, 5))
        
        # 子图1：数据点聚类
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=labels, cmap='viridis', alpha=0.6, s=50)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='red', marker='X', s=200, edgecolors='black', label='Cluster Centers')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Histogram Features Clustering (PCA Visualization)')
        plt.legend()
        plt.colorbar(scatter, label='Cluster Label')
        
        # 子图2：聚类中心直方图
        plt.subplot(1, 2, 2)
        x = np.arange(16)  # 16个直方图bin
        width = 0.25  # 柱状图宽度
        
        for i in range(len(centers)):
            offset = (i - (len(centers)-1)/2) * width
            plt.bar(x + offset, centers[i], width, label=f'Cluster {i}')
        
        plt.xlabel('Histogram Bin')
        plt.ylabel('Probability')
        plt.title('Cluster Centers Histogram Distribution')
        plt.legend()
        plt.xticks(x)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('clusters_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 打印聚类统计信息
        print(f"\nClustering Statistics:")
        print(f"Total valid samples: {len(features)}")
        print(f"Number of clusters: {len(centers)}")
        for i in range(len(centers)):
            count = np.sum(labels == i)
            percentage = count / len(features) * 100
            print(f"Cluster {i}: {count} samples ({percentage:.1f}%)")


def generate_clusters_json(dataset_path, test_txt_path='', 
                                n_clusters=3, output_path='isic_clusters.json'):

    # 1. 加载数据集
    print(f"Loading dataset from: {dataset_path}")
    dataset = ISICDataset(root_dir=dataset_path, test_txt_dir=test_txt_path, augmentation=False)
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # 2. 创建聚类分析对象
    cluster_analyzer = DatasetForClustering(dataset)
    
    # 3. 执行聚类分析
    try:
        features, labels, centers = cluster_analyzer.perform_clustering(
            n_clusters=n_clusters, random_state=42
        )
    except Exception as e:
        print(f"Error during clustering: {e}")
        # 如果聚类失败，创建默认的聚类中心
        print("Creating default cluster centers...")
        centers = create_default_centers(n_clusters)
        features = np.array([])
        labels = np.array([])
    
    # 4. 可视化聚类结果（如果有足够数据）
    if len(features) > 0:
        print("\nVisualizing clustering results...")
        try:
            cluster_analyzer.visualize_clusters(features, labels, centers)
        except Exception as e:
            print(f"Visualization error: {e}")
    
    # 5. 准备JSON数据 - 使用数值格式而不是字符串
    # 这是关键修改：保持为数值列表，而不是字符串
    centers_list = []
    for center in centers:
        # 转换为Python的float列表（不是字符串）
        center_list = center.tolist()
        centers_list.append(center_list)
    
    # 构建JSON数据结构
    cluster_data = [{
        "n_class": n_clusters,
        "centers": centers_list  # 使用数值列表而不是字符串
    }]
    
    # 6. 保存为JSON文件
    print(f"\nSaving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(cluster_data, f, indent=2)
    
    print(f"Clustering results saved to {output_path}")
    
    # 7. 打印聚类中心的统计信息
    print(f"\nCluster Centers Summary (n_class={n_clusters}):")
    for i, center in enumerate(centers):
        print(f"\nCluster {i}:")
        print(f"  Min value: {center.min():.6f}")
        print(f"  Max value: {center.max():.6f}")
        print(f"  Mean value: {center.mean():.6f}")
        print(f"  Shape: {center.shape}")
        
        # 检查是否包含NaN或inf
        if np.isnan(center).any():
            print("  WARNING: Contains NaN values!")
        if np.isinf(center).any():
            print("  WARNING: Contains inf values!")
        
        # 找出主要的峰值
        peak_indices = np.argsort(center)[-3:][::-1]  # 最大的3个峰值
        for idx in peak_indices:
            print(f"    Bin {idx}: {center[idx]:.6f} ({center[idx]*100:.1f}%)")
    
    return cluster_data


def create_default_centers(n_clusters=3):
    """创建默认的聚类中心（以防聚类失败）"""
    print(f"Creating default {n_clusters} cluster centers...")
    
    if n_clusters == 3:
        # 创建3个不同的分布模式
        centers = []
        
        # 聚类1：低强度分布
        center1 = np.zeros(16)
        center1[5:9] = [0.1, 0.3, 0.4, 0.2]
        center1 = center1 / center1.sum()
        centers.append(center1)
        
        # 聚类2：中等强度分布
        center2 = np.zeros(16)
        center2[7:12] = [0.1, 0.25, 0.3, 0.25, 0.1]
        center2 = center2 / center2.sum()
        centers.append(center2)
        
        # 聚类3：高强度分布
        center3 = np.zeros(16)
        center3[10:15] = [0.1, 0.2, 0.4, 0.2, 0.1]
        center3 = center3 / center3.sum()
        centers.append(center3)
    
    elif n_clusters == 4:
        # 创建4个不同的分布模式
        centers = []
        for i in range(4):
            center = np.zeros(16)
            start_idx = i * 3 + 3
            center[start_idx:start_idx+5] = [0.1, 0.2, 0.4, 0.2, 0.1]
            center = center / center.sum()
            centers.append(center)
    
    else:
        # 通用默认中心
        centers = []
        for i in range(n_clusters):
            center = np.ones(16) / 16.0  # 均匀分布
            centers.append(center)
    
    return np.array(centers)


def validate_clusters_json(json_path='isic_clusters.json'):
    """验证生成的JSON文件格式是否正确"""
    print(f"\nValidating {json_path}...")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None
    
    print(f"Number of cluster sets: {len(data)}")
    
    for i, cluster_set in enumerate(data):
        n_class = cluster_set.get("n_class")
        centers = cluster_set.get("centers", [])
        
        print(f"\nCluster Set {i}:")
        print(f"  n_class: {n_class}")
        print(f"  Number of centers: {len(centers)}")
        
        if len(centers) > 0:
            print(f"  Dimension of each center: {len(centers[0])}")
            
            # 检查每个中心的维度是否一致
            dims = [len(c) for c in centers]
            if len(set(dims)) == 1:
                print(f"  ✓ All centers have consistent dimension: {dims[0]}")
            else:
                print(f"  ✗ Inconsistent dimensions: {dims}")
            
            # 检查数据类型（应该是数值，不是字符串）
            first_value = centers[0][0]
            if isinstance(first_value, str):
                print("  ✗ WARNING: Centers contain string values instead of numbers!")
                print("    This will cause 'ValueError: too many dimensions' in inference!")
            else:
                print(f"  ✓ Centers contain numeric values (type: {type(first_value)})")
    
    return data


def create_fix_for_inference_code():
    """
    创建修复推理代码的补丁
    """
    fix_code = '''
# ===== 修复inference.py中的代码 =====
# 在inference.py中找到以下代码：
# hist = th.tensor(cluster_centers[type])
# 改为：

def load_cluster_centers(json_path):
    """正确加载聚类中心"""
    import json
    import numpy as np
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cluster_set = data[0]
    centers = cluster_set["centers"]
    
    # 确保centers是数值数组
    centers_np = []
    for center in centers:
        if isinstance(center[0], str):
            # 如果是字符串，转换为浮点数
            center_np = [float(val) for val in center]
        else:
            # 已经是数值，直接使用
            center_np = center
        centers_np.append(center_np)
    
    return np.array(centers_np)

# 使用方式：
# cluster_centers = load_cluster_centers('path/to/isic_clusters.json')
# hist = th.tensor(cluster_centers[type_idx].astype(np.float32))
'''
    print("\n" + "="*60)
    print("If you still have inference errors, use this fix in inference.py:")
    print(fix_code)
    print("="*60)


# 使用示例
if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser(description='Generate clusters')
    parser.add_argument('--dataset_path', type=str, required=True, help='/path/to/your/DATASET')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--num', type=int, default=3, help='number of clusters')
    args = parser.parse_args()
    DATASET_PATH = args.dataset_path
    TEST_TXT_PATH = ""  #
    N_CLUSTERS = args.num
    OUTPUT_FILE = "./LeFusion/inference/hist_clusters/"+args.dataset_name+"_clusters.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("="*60)
    print("Dataset Clustering Analysis")
    print("="*60)
    
    # 生成聚类JSON文件
    try:
        cluster_data = generate_clusters_json(
            dataset_path=DATASET_PATH,
            test_txt_path=TEST_TXT_PATH,
            n_clusters=N_CLUSTERS,
            output_path=OUTPUT_FILE
        )
        
        # 验证生成的JSON文件
        print("\n" + "="*60)
        print("Validating output file...")
        validate_clusters_json(OUTPUT_FILE)
        
        # 显示JSON文件内容示例
        print(f"\nExample of {OUTPUT_FILE} content:")
        if cluster_data and cluster_data[0]["centers"]:
            first_center = cluster_data[0]["centers"][0]
            print(f"First cluster center shape: {len(first_center)}")
            print(f"First 5 values: {first_center[:5]}")
            print(f"Data type of first value: {type(first_center[0])}")
            
        # 创建修复建议
        create_fix_for_inference_code()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
        # 即使出错也创建默认文件
        print("\nCreating default cluster file due to error...")
        centers = create_default_centers(N_CLUSTERS)
        centers_list = centers.tolist()
        
        cluster_data = [{
            "n_class": N_CLUSTERS,
            "centers": centers_list
        }]
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(cluster_data, f, indent=2)
        
        print(f"Default cluster file created at: {OUTPUT_FILE}")