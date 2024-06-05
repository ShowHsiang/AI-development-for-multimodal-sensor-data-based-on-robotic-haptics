import os
import pandas as pd
import glob

# 读取 Excel 文件
description_df = pd.read_excel('dataset_description1.xlsx', usecols=['Unnamed: 0', 'Unnamed: 2'])

# 提取布料编号和名称
description_df['Unnamed: 0'] = description_df['Unnamed: 0'].str.extract(r'(\d+)#')[0].astype(int)
sample_to_name = dict(zip(description_df['Unnamed: 0'], description_df['Unnamed: 2']))

# 获取 ctl_130-140 文件夹中的所有 CSV 文件
csv_files = glob.glob('dataset/Pressure_Texture_dataset_csv/ctl_130-140/*.csv')

# 遍历 CSV 文件
for csv_file in csv_files:
    # 从文件名中提取布料编号
    file_name = os.path.basename(csv_file)
    sample_number = int(file_name.split('S')[1].split('_')[0].split('-')[0])

    # 获取布料名称
    if sample_number in sample_to_name:
        label = sample_to_name[sample_number]
        
        # 读取 CSV 文件并添加布料名称列
        df = pd.read_csv(csv_file)
        df['label'] = label
        
        # 保存修改后的 CSV 文件
        df.to_csv(csv_file, index=False)
        print(f'Added label to {csv_file}')

print('All matching CSV files have been updated.')
