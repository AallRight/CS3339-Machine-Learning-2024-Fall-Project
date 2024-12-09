import numpy as np

def load_and_inspect_npy(file_path):
    # 加载 .npy 文件
    data = np.load(file_path, allow_pickle=True)  # 如果是对象数据，可以使用 allow_pickle=True
    
    # 输出数据的基本信息
    print(f"数据类型: {type(data)}")
    print(f"数据形状: {data.shape}")
    print(f"数据类型（数据元素类型）: {data.dtype}")
    
    # 如果数据是多维数组，可以查看前几个元素
    print("前几个数据元素:")
    print(data[1000:1010])  # 输出前五个元素，调整索引根据数据类型
    
    # 使用集合来记录不同的值
    unique_values = set(data)  # set 会自动去重

    # 将集合转为列表
    unique_values_list = list(unique_values)
    
    print(f"数据中不同的值数量: {len(unique_values_list)}")
    print(f"不同的值: {unique_values_list}")
    
# 替换为你的 .npy 文件路径
npy_file_path = './data/train_labels.npy'
load_and_inspect_npy(npy_file_path)
