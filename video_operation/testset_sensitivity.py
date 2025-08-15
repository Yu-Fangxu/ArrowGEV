import argparse
import asyncio
import json
import os
import re
from typing import Any, Dict, List

from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm.asyncio import tqdm_asyncio
from string import Template
# --- Prompt模板 ---
# 这个模板和之前一样，它明确地要求模型输出JSON格式，这对后续解析很有帮助。
RM_PROMPT_TEMPLATE = """
**You are an AI assistant specializing in the analysis of temporal properties of events.**
You will be given a sentence describing an event.

**Your task is to:**
1.  Analyze the event described in the sentence.
2.  Determine if the event is temporally **sensitive** or **insensitive**.
3.  Output the results in a strict JSON format without any additional text or explanations.

---
### **Input**
**Event Sentence:** ${sentence}

---
### **Evaluation Criteria**
* **Time-Sensitive (sensitive: yes):** The event has a clear forward direction. If played in reverse, it describes a different, often nonsensical or opposite, event. This indicates temporal asymmetry.
    * *Example:* "A person puts a picture on the wall." (Reversed: "A person takes a picture off the wall.")
    * *Example:* "A glass shatters." (Reversed: "Shards of glass assemble into a whole glass.")

* **Time-Insensitive (sensitive: no):** The event is a continuous state or a cyclical action. If played in reverse, the fundamental nature of the event does not change. This indicates temporal symmetry.
    * *Example:* "A person is playing with a light switch." (Reversed: Still looks like a person playing with a light switch.)
    * *Example:* "A ball is bouncing in place." (Reversed: Still a ball bouncing in place.)
---
### **Output Format**
Now, please output your result below in a JSON format by filling in the placeholders in [] without any explanations:

{
  "reason": "[Briefly explain why the event is time-sensitive or time-insensitive, describing the forward and reverse action.]",
  "sensitive": "[yes/no]"
}
"""

def parse_llm_output(output_string: str) -> str:
    """
    健壮地解析LLM的输出，以提取 "sensitive" 的值。
    优先尝试解析完整的JSON对象，如果失败则使用正则表达式作为后备。
    """
    try:
        # 使用 OpenAI 的 JSON Mode 后，输出本身就是有效的JSON字符串
        data = json.loads(output_string)
        sensitive_value = data.get("sensitive", "unknown").lower()
        if sensitive_value in ["yes", "no"]:
            return sensitive_value
    except (json.JSONDecodeError, AttributeError):
        # 如果JSON解析失败（例如模型没有完全遵循指令），则使用正则
        match = re.search(r'"sensitive"\s*:\s*"(yes|no)"', output_string, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return "unknown" # 如果所有方法都失败了


async def process_item_async(
    client: AsyncOpenAI,
    item: Dict[str, Any],
    model: str,
    semaphore: asyncio.Semaphore,
    input_file_path: str
) -> Dict[str, Any]:
    """
    异步处理单个JSON对象：发送给OpenAI并解析结果。
    """
    # 使用信号量控制并发数量，防止触发速率限制
    async with semaphore:
        if "charades" in input_file_path or "train_2k5" in input_file_path:
            sentence = item.get("sentence")
        elif "tvgbench" in input_file_path:
            sentence = item.get("question")
        elif "activitynet" in input_file_path:
            pass
        if not sentence:
            item["sensitive"] = "no_sentence"
            return item

        # prompt = RM_PROMPT_TEMPLATE.format(sentence=sentence)
        prompt = Template(RM_PROMPT_TEMPLATE).substitute(sentence=sentence)
        # print(prompt)
        for attempt in range(3): # 最多重试3次
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, # 对于分类任务，温度设为0以获得确定性结果
                    response_format={"type": "json_object"}, # 强制模型输出JSON格式
                )
                
                llm_output = response.choices[0].message.content
                # print(llm_output)
                sensitive_value = parse_llm_output(llm_output)
                
                # 创建一个新的字典以保持原始数据不变
                result_item = item.copy()
                result_item["sensitive"] = sensitive_value
                return result_item

            except RateLimitError:
                wait_time = (2 ** attempt) * 2 # 指数退避：2s, 4s, 8s
                print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                await asyncio.sleep(wait_time)
            except APIError as e:
                print(f"API Error occurred: {e}. Retrying...")
                await asyncio.sleep(2)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break # 其他未知错误，不再重试
    
    # 如果所有尝试都失败了
    result_item = item.copy()
    result_item["sensitive"] = "error"
    return result_item


async def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(
        description="Analyze sentence time-sensitivity using the OpenAI API."
    )
    parser.add_argument(
        "-i", "--input-file", default="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/timer1/annotations/train_2k5.json", help="Path to the input JSON file."
    )
    parser.add_argument(
        "-o", "--output-file", default="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/timer1/annotations/train_2k5_sens.json", help="Path to the output JSON file."
    )
    parser.add_argument(
        "-m", "--model", default="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/zhenzcao/img_translate/2.MLLM/3.pretrained_models/Qwen2.5-VL-72B-Instruct_new/Qwen2.5-VL-72B-Instruct/", help="OpenAI model to use (e.g., gpt-4o, gpt-3.5-turbo)."
    )
    parser.add_argument(
        "-c", "--max-concurrency", type=int, default=10, help="Maximum number of concurrent API requests."
    )
    parser.add_argument(
        "--api-key", default="EMPTY", help="OpenAI API key (overrides environment variable)."
    )
    args = parser.parse_args()

    # 初始化OpenAI客户端
    api_key = args.api_key 
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or use the --api-key argument.")
    openai_ip = "29.226.50.232"
    port = "8000"
    base_url=f"http://{openai_ip}:{port}/v1"
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # 读取输入文件
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data_to_process = json.load(f)
        if not isinstance(data_to_process, list):
             raise TypeError("Input JSON must be a list of objects.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error reading or parsing JSON file: {e}")
        return

    # 创建并发控制器和任务列表
    semaphore = asyncio.Semaphore(args.max_concurrency)
    tasks = [
        process_item_async(client, item, args.model, semaphore, args.input_file)
        for item in data_to_process
    ]

    print(f"Starting analysis on {len(data_to_process)} items using model '{args.model}'...")
    
    # 使用tqdm_asyncio.gather来执行所有任务并显示进度条
    results = await tqdm_asyncio.gather(*tasks)
    
    # 过滤掉可能出现的None值（虽然目前逻辑下不会出现）
    final_results = [r for r in results if r is not None]

    # 保存结果到输出文件
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())