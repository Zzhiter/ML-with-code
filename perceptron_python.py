import numpy as np
import time


def loadData(filename):
    """
    Args:
        filename: Mnist数据集的存储路径

    Returns:返回数据集的特征向量和标签类别

    """
    data_list = []
    label_list = []

    with open(filename, "r") as f:
        for line in f.readlines():
            curline = line.split().split(",")
            data_list.append([int(feature) for feature in curline[1:]])
            if int(curline[0]) >= 5:
                label_list.append(1)
            else:
                label_list.append(-1)

    # 将list类型的特征向量，变化成矩阵，维度为（60000， 784）
    data_matrix = np.array(data_list)
    # 将list类型的标签类别，变化成矩阵，维度为（1， 60000）
    label_matrix = np.array(label_list)

    return data_matrix, label_matrix


class Perceptron(object):
    """
    定义Perception类，包含算法的原始形式和对偶性质的实现函数
    """

    def __init__(self, data_matrix, label_matrix, iteration=30, learning_rate=0.0001):
        """
        Args:
            data_matrix: 数据的特征向量
            label_matrix: 数据的标签类别
            iteration: 迭代次数
            learning_rate: 学习率
        """
        self.data_martix = data_matrix
        self.label_martix = label_matrix
        self.iteraion = iteration
        self.learning_rate = learning_rate

    def original_method(self):
        """
        感知机学习算法的原始形式的实现
        Returns: 返回参数w，b
        """

        data_matrix = self.data_martix
        label_matrix = self.label_martix
        #
        input_num, feature_num = np.shape(data_matrix)
        print("训练集的维度为" + data_matrix.shape)

        # 参数初始化
        w = np.random.randn(1, feature_num)
        b = np.random.randn()
        # 迭代iteration次
        for iter in range(self.iteration):
            # 在每个样本上进行判断
            for i in range(input_num):
                x_i = data_matrix[i]
                y_i = label_matrix[i]
                result = y_i * (np.matmul(w, x_i.T) + b)
                if result <= 0:
                    w = w + self.learning_rate * y_i * x_i
                    b = b + self.learning_rate * y_i
            print(f'this is {iter} round, the '
                  f'total round is {self.iteration}')
        assert (w.shape == (1, feature_num)), "error"
        return w, b

    def dual_method(self):
        '''
        感知机学习算法的对偶形式的实现
        :return:
        '''
        data_matrix = self.data_matrix
        label_matrix = self.label_matrix.T

        # input_num表示训练集数目，feature_num表示特征数目
        input_num, feature_num = np.shape(data_matrix)
        # 系数a，初始化为全0的(1,input_num)的矩阵
        a = np.zeros(input_num)
        # 系数b，初始化为0
        b = 0

        # 计算出gram矩阵
        print(a.shape)
        gram = np.matmul(data_matrix, data_matrix.T)
        assert (gram.shape == (input_num, input_num))
        print(gram.shape)

        # 迭代iteration次
        for iter in range(self.iteration):
            # 在每一个样本上都进行判断
            for i in range(input_num):
                result = 0
                for j in range(input_num):
                    result += a[j] * label_matrix[j] * gram[j][i]
                result += b
                result *= label_matrix[i]

                # 判断当前样本会不会被误分类
                if (result <= 0):
                    a[i] = a[i] + self.learning_rate
                    b = b + self.learning_rate * label_matrix[i]
                else:
                    continue

            print(f"this is {iter} round,the total round is {self.iteration}.")

        w = np.matmul(np.multiply(a, self.label_matrix), self.data_matrix)
        print(w.shape)

        return w, b


def test_model(test_data_matrix, test_label_matrix, w, b):
    """
    Args:
        test_data_matrix: 测试集特征向量
        test_label_matrix: 测试集标签类别
        w: 计算的到的w
        b: 计算得到的b

    Returns: 返回test准确率

    """

    test_input_num, _ = np.shape(test_data_matrix)
    error_num = 0

    # 统计在测试集合上被错误分类的个数
    for i in range(test_input_num):
        result = (test_label_matrix[i]) * (np.matmul(w, test_data_matrix[i].T) + b)
        if (result <= 0):
            error_num += 1
        else:
            continue

    accuracy = (test_input_num - error_num) / test_input_num
    return accuracy


def main():
    start = time.time()

    train_data_list, train_label_list = loadData("../MnistData/mnist_train.csv")
    test_data_list, test_label_list = loadData("../MnistData/mnist_test.csv")
    print("finished load data.")
    perceptron = Perceptron(train_data_list[:1000], train_label_list[:1000], iteration=30, learning_rate=0.0001)
    w, b = perceptron.dual_method()

    accuracy = test_model(test_data_list, test_label_list, w, b)
    # 获取当前时间，作为程序结束运行时间
    end = time.time()

    # 打印模型在测试集上的准确率
    print(f"accuracy is {accuracy}.")
    # 打印程序运行总时间
    print(f"the total time is {end - start}.")
