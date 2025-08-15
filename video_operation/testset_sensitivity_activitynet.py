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
  "reason": "[Briefly explain why the event is reversible or irreversible, describing the forward and reverse action.]",
  "sensitive": "[yes/no]"
}
"""

def parse_llm_output(output_string: str) -> str:
    """健壮地解析LLM的输出，以提取 "sensitive" 的值。"""
    try:
        data = json.loads(output_string)
        sensitive_value = data.get("sensitive", "unknown").lower()
        if sensitive_value in ["yes", "no"]:
            return sensitive_value
    except (json.JSONDecodeError, AttributeError):
        match = re.search(r'"sensitive"\s*:\s*"(yes|no)"', output_string, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return "unknown"


async def process_sentence_async(
    client: AsyncOpenAI,
    item_to_process: Dict[str, Any],
    model: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    异步处理单个句子，并返回带有上下文(video_id, index)和结果的字典。
    """
    async with semaphore:
        sentence = item_to_process.get("sentence")
        if not sentence:
            item_to_process["sensitive"] = "no_sentence"
            return item_to_process

        # prompt = RM_PROMPT_TEMPLATE.format(sentence=sentence)
        prompt = Template(RM_PROMPT_TEMPLATE).substitute(sentence=sentence)
        # print(prompt)
        for attempt in range(3): # 最多重试3次
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                llm_output = response.choices[0].message.content
                sensitive_value = parse_llm_output(llm_output)
                item_to_process["sensitive"] = sensitive_value
                return item_to_process

            except RateLimitError:
                wait_time = (2 ** attempt) * 2
                print(f"Rate limit hit. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            except APIError as e:
                print(f"API Error on sentence '{sentence[:30]}...': {e}. Retrying...")
                await asyncio.sleep(2)
            except Exception as e:
                print(f"Unexpected error on sentence '{sentence[:30]}...': {e}")
                break
    
    item_to_process["sensitive"] = "error"
    return item_to_process


async def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(
        description="Analyze sentence time-sensitivity using the OpenAI API."
    )
    parser.add_argument(
        "-i", "--input-file", default="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/charades/Charades_anno/Charades_sta_test.json", help="Path to the input JSON file."
    )
    parser.add_argument(
        "-o", "--output-file", default="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/charades/Charades_anno/Charades_sta_test_sens.json", help="Path to the output JSON file."
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

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        if not isinstance(input_data, dict):
             raise TypeError("Input JSON must be a dictionary of video objects.")
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        print(f"Error reading or parsing JSON file: {e}")
        return

    # --- 2. 扁平化数据结构以便并发处理 ---
    # 将嵌套的句子转换为一个扁平的任务列表，每个任务都包含其来源信息。
    items_to_process = []
    for video_id, video_data in input_data.items():
        if "sentences" in video_data and isinstance(video_data["sentences"], list):
            for i, sentence in enumerate(video_data["sentences"]):
                items_to_process.append({
                    "video_id": video_id,
                    "sentence_index": i,
                    "sentence": sentence
                })
    
    if not items_to_process:
        print("No sentences found to process in the input file.")
        return

    # --- 3. 异步执行分析任务 ---
    semaphore = asyncio.Semaphore(args.max_concurrency)
    tasks = [
        process_sentence_async(client, item, args.model, semaphore)
        for item in items_to_process
    ]
    print(f"Starting analysis on {len(items_to_process)} total sentences from {len(input_data)} videos...")
    
    processed_results = await tqdm_asyncio.gather(*tasks)

    # --- 4. 重组数据到最终格式 ---
    # 将扁平化的结果重新组合回原始的嵌套结构中。
    output_data = input_data.copy() # 复制原始数据以保留所有字段
    for video_id, video_data in output_data.items():
        num_sentences = len(video_data.get("sentences", []))
        # 初始化一个占位符列表
        video_data["sensitive"] = ["unprocessed"] * num_sentences
    
    for result in processed_results:
        video_id = result["video_id"]
        index = result["sentence_index"]
        sensitive_value = result["sensitive"]
        # 根据索引将结果填入正确位置
        if video_id in output_data and index < len(output_data[video_id]["sensitive"]):
            output_data[video_id]["sensitive"][index] = sensitive_value

    # --- 5. 保存结果 ---
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Analysis complete. Results saved to {args.output_file}")
    print(f"Note: The sentence timestamps are preserved in the original 'timestamps' key for each video.")

if __name__ == "__main__":
    asyncio.run(main())