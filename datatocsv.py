# CUDA_VISIBLE_DEVICES=0 python datatocsv.py
import os
import re
import pandas as pd

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract headers
    headers = lines[0].strip().split()
    headers.remove('FA')
    headers.remove('PRES')
    
    tax_list = []
    adc_list = []
    ts_list = []

    # Process lines and extract data
    for line in lines[1:]:
        matches = re.findall(r'\[([^\]]*)\]', line)
        if len(matches) == 4:
            tax = matches[1].strip()
            adc = matches[2].strip()
            ts = line.split()[-1].strip()  # Extract the timestamp
            
            tax_list.append(tax)
            adc_list.append(adc)
            ts_list.append(ts)
    
    data = {
        'TAX': tax_list,
        'ADC': adc_list,
        'TS': ts_list
    }
    df = pd.DataFrame(data)
    
    return df

def convert_files_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if not filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            df = process_file(file_path)
            csv_file_path = os.path.join(output_directory, filename + '.csv')
            df.to_csv(csv_file_path, index=False)
            print(f'Converted {filename} to {csv_file_path}')

if __name__ == "__main__":
    input_directory = 'dataset/Pressure_Texture_dataset_raw/20240521 CTL101-129'  # Replace with your input directory
    output_directory = 'dataset/Pressure_Texture_dataset_csv/ctl_101-129'  # Replace with your output directory
    convert_files_in_directory(input_directory, output_directory)
