import pickle

def load_and_inspect_pkl(file_path):
    # 尝试加载 pkl 文件
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return

    # 输出数据的类型
    print(f"数据类型: {type(data)}")

    # 如果数据是字典类型，打印键
    if isinstance(data, dict):
        print("数据是字典，键如下:")
        for key in data.keys():
            print(f"  - {key}")
        # 可以根据需要进一步检查某个具体的键的值
        print("\n字典的一个样例值:")
        print(data[list(data.keys())[0]])  # 打印第一个键对应的值

    # 如果数据是列表或数组类型，打印前几个元素
    elif isinstance(data, list):
        print("数据是列表，前几个元素如下:")
        print(data[:1])  # 打印前五个元素

    # 如果数据是 NumPy 数组类型
    elif isinstance(data, (list, tuple)) and hasattr(data, 'shape'):
        print("数据是数组或类似数组的结构，数组形状:")
        print(data.shape)

    # 如果数据是其他类型，直接打印数据的摘要
    else:
        print("数据内容:")
        print(data)

# 替换为你的 .pkl 文件路径
pkl_file_path = 'data/train_feature_dense.pkl'
model_file_path = 'model/method2/svm_model_20241207_232638.pkl'
load_and_inspect_pkl(model_file_path)
