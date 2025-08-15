import json

# 定义输入和输出文件名
# input_filename = '/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/charades/Charades_anno/Charades_sta_test_sens.json'
# output_filename = '/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/charades/Charades_anno/Charades_sta_test_only_insens.json'
input_filename = '/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/activitynet/annotations/sentence_temporal_grounding/test_sens.json'
output_filename = '/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/activitynet/annotations/sentence_temporal_grounding/test_only_insens.json'

# 加载输入JSON数据
with open(input_filename, 'r') as f:
    data = json.load(f)

# 创建一个字典来保存过滤后的结果
filtered_output = {}

# 遍历输入中的每个视频ID和其数据
for video_id, video_data in data.items():
    
    # 用于存储当前视频过滤后项目的列表
    new_timestamps = []
    new_sentences = []
    # 我们也可以过滤sensitive列表本身，只保留'yes'
    new_sensitive_list = []

    # 使用enumerate来同时获取索引和值
    for i, sensitive_flag in enumerate(video_data.get('sensitive', [])):
        if sensitive_flag == 'no':
            # 如果是 'yes'，则添加对应的条目
            new_timestamps.append(video_data['timestamps'][i])
            new_sentences.append(video_data['sentences'][i])
            new_sensitive_list.append(sensitive_flag)
            
    # 只有当我们为此视频找到敏感内容时，才将其添加到输出中
    if new_timestamps:
        # 创建一个输出视频对象
        output_video_data = {}
        # 复制原始视频中的所有键值对
        for key, value in video_data.items():
            output_video_data[key] = value
        
        # 用过滤后的列表更新（替换）键
        output_video_data['timestamps'] = new_timestamps
        output_video_data['sentences'] = new_sentences
        output_video_data['sensitive'] = new_sensitive_list
        
        # 将更新后的视频数据添加到最终输出中
        filtered_output[video_id] = output_video_data

# 将过滤后的数据写入输出JSON文件
with open(output_filename, 'w') as f:
    json.dump(filtered_output, f, indent=4)

print(f"✅ 完整过滤后的数据已保存到 '{output_filename}'")