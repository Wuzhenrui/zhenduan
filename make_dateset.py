# -*-coding:utf-8-*-
# 在已经将图片分好类的情况下 给图片加上标签
# 好像就是生成一个txt  里面有图片路径+名字  和对应的标签
#使用的时候   修改get_files_list里面的分类  再修改 main里面的路径即可
import os
import os.path


def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def get_files_list(dir):
    '''
    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
    global labels
    files_list = []  # 写入文件的数据
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            print("parent is: " + parent)
            print("filename is: " + filename)
            print(os.path.join(parent, filename).replace('\\', '/'))  # 输出rootdir路径下所有文件（包含子文件）信息
            curr_file = parent.split(os.sep)[-1]  # 获取正在遍历的文件夹名（也就是类名）
            # 根据class名确定labels-----------------------------------------------------------------------------------------
            if curr_file == "normal":
                labels = 0
            elif curr_file == "3_fault":
                labels = 1
            elif curr_file == "4_fault":
                labels = 2
            elif curr_file == "8_fault":
                labels = 3
            elif curr_file == "19_fault":
                labels = 4

            dir_path = parent.replace('\\', '/').split('/')[-2]  # train?val?test?

            curr_file = os.path.join(dir_path, curr_file)  # 相对路径

            files_list.append([os.path.join(curr_file, filename).replace('\\', '/'), labels])  # 相对路径+label

            # 写入csv文件
            path = "%s" % os.path.join(curr_file, filename).replace('\\', '/')
            label = "%d" % labels
            list = [path, label]
            data = pd.DataFrame([list])
            # if dir == 'E:/桌面/zhenduan/images/training':
            #     data.to_csv("E:/桌面/zhenduan/images/training.csv", mode='a', header=False, index=False)
            # elif dir == 'E:/桌面/zhenduan/images/testing':
            #     data.to_csv("E:/桌面/zhenduan/images/testing", mode='a', header=False, index=False)
            if dir == './Dataset/train':
                data.to_csv("./Dataset/train.csv", mode='a', header=False, index=False)
            elif dir == './Dataset/val':
                data.to_csv("./Dataset/val.csv", mode='a', header=False, index=False)

    return files_list


if __name__ == '__main__':
    import pandas as pd

    # 先生成两个csv文件夹
    # DataFrame  设计一个表格  这里是   表格中有  path  和  label
    df = pd.DataFrame(columns=['path', 'label'])
    # to_csv  是DataFrame类的一个函数  将df 保存在该路径下   index=False  表示不保留索引  （这样表格前面不会有序号）
    #df.to_csv("./Dataset/train.csv", index=False)
    df.to_csv("E:/桌面/zhenduan/images/training.csv", index=False)

    df2 = pd.DataFrame(columns=['path', 'label'])
    #df2.to_csv("./Dataset/val.csv", index=False)
    df2.to_csv("E:/桌面/zhenduan/images/testing.csv", index=False)

    # 写入txt文件
    # train_dir = './Dataset/train'
    train_dir = 'E:/桌面/zhenduan/images/training'
    train_txt = 'E:/桌面/zhenduan/images/training.txt'
    train_data = get_files_list(train_dir)
    write_txt(train_data, train_txt, mode='w')

    # val_dir = './Dataset/val'
    # val_txt = './Dataset/val.txt'
    val_dir = 'E:/桌面/zhenduan/images/testing'
    val_txt = 'E:/桌面/zhenduan/images/testing.txt'
    val_data = get_files_list(val_dir)
    write_txt(val_data, val_txt, mode='w')
