import json
import os
# 读取JSON文件
input_json_file = '/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/train_v4_cloud_simple_all.json'  # 这里是你的JSON文件路径
output_json_file = '/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/train_v4_cloud_simple_all.json'  # 输出修改后的JSON文件路径

# 设定要拼接的基路径
base_path = "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/"

# 读取原始JSON文件
with open(input_json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每个问题并更新path字段
for item in data:
    if 'video' in item:
        # 拼接路径
        path = item['video']
        # path = path.replace("_reverse", "")
        # item['video'] = path
        
        name, extension = os.path.splitext(path)
        new_path =name + '_reverse_only' + extension
        item['video_reverse_path'] = new_path
        # new_path =name + '_reverse' + extension
        # item['video'] = new_path
        # item['video'] = os.path.join(base_path, item['video'].lstrip('./'))
        # print(new_path)
        # break
# 将修改后的数据写入到新的JSON文件
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("路径更新完成，新的JSON文件已保存到", output_json_file)
