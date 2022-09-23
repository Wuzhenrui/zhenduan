# -*-coding:utf-8-*-


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
    files_list = [] #写入文件的数据
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            print("parent is: " + parent)
            print("filename is: " + filename)
            print(os.path.join(parent, filename).replace('\\','/'))  # 输出rootdir路径下所有文件（包含子文件）信息
            curr_file = parent.split(os.sep)[-1]	#获取正在遍历的文件夹名（也就是类名）
			#根据class名确定labels
            if curr_file == "class1":
                labels = 0
            elif curr_file == "class1":
                labels = 1
            elif curr_file == "class1":
                labels = 2

       		dir_path = parent.replace('\\', '/').split('/')[-2]   #train?val?test?

            curr_file = os.path.join(dir_path, curr_file)  #相对路径

            files_list.append([os.path.join(curr_file, filename).replace('\\','/'), labels])	#相对路径+label


    		#写入csv文件
            path = "%s" % os.path.join(curr_file, filename).replace('\\','/')
            label = "%d" % labels
            list = [path, label]
            data = pd.DataFrame([list])
            if dir == './Dataset/train':
                data.to_csv("./Dataset/train.csv", mode='a', header=False, index=False)
            elif dir == './Dataset/val':
                data.to_csv("./Dataset/val.csv", mode='a', header=False, index=False)

    return files_list


if __name__ == '__main__':

    import pandas as pd
    #先生成两个csv文件夹
    df = pd.DataFrame(columns=['path', 'label'])
    df.to_csv("./Dataset/train.csv", index=False)

    df2 = pd.DataFrame(columns=['path', 'label'])
    df2.to_csv("./Dataset/val.csv", index=False)

    #写入txt文件
    train_dir = './Dataset/train'
    train_txt = './Dataset/train.txt'
    train_data = get_files_list(train_dir)
    write_txt(train_data, train_txt, mode='w')

    val_dir = './Dataset/val'
    val_txt = './Dataset/val.txt'
    val_data = get_files_list(val_dir)
    write_txt(val_data, val_txt, mode='w')


