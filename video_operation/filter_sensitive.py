import json

# 定义输入和输出文件名
input_filename = '/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/timer1/annotations/tvgbench_sense.json'
output_filename = '/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/timer1/annotations/tvgbench_only_insens.json'

# 创建一个示例 input.json 文件
# 在您的实际使用中，您将拥有自己的 input.json 文件

# with open(input_filename, 'w') as f:
#     json.dump(input_data, f, indent=2)

# 读取JSON文件
with open(input_filename, 'r') as f:
    data = json.load(f)

# 过滤数据，只保留 sensitive == 'yes' 的条目
sensitive_data = [item for item in data if item.get('sensitive') == 'no']

# 将过滤后的数据写入新的JSON文件
with open(output_filename, 'w') as f:
    json.dump(sensitive_data, f, indent=4)

print(f"Filtered data has been saved to '{output_filename}'")