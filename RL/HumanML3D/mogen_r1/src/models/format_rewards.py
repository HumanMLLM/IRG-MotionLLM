import re
import os
from datetime import datetime


# def format_reward_mogen_thinking(completions, **kwargs):
#     """Check if the Qwen model output matches a specific format."""

#     pattern = r"<think>.*?[analysis].*?[/analysis]\s*[planning].*?[planning]\s*(?:\[self-verification-placeholder]\s*)+</think>*?"
#     self_verification_pattern = r"[generation]\s*<motion>\s*(?=.*?(?:<motion_\d>\s*){2,})\s*</motion>\s*[/generation]\s*[assessment].*?[/assessment]\s*[planning].*?[/planning]"
#     pattern = pattern.replace("[self-verification-placeholder]", self_verification_pattern)
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#     if os.getenv("DEBUG_MODE") == "true":
#         log_path = os.getenv("LOG_PATH")
#         with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
#             f.write(f"------------- {current_time} Format reward -------------\n")
#             for content, match in zip(completion_contents, matches):
#                 f.write(f"Content: {content}\n")
#                 f.write(f"Has format: {bool(match)}\n")
#     return [1.0 if match else 0.0 for match in matches]
def format_reward_mogen_strict(completions, pattern=r"<think>\s*\[plan\].*?\[/plan\]\s*\[analyze\].*?\[/analyze\]\s*\[plan\].*?\[/plan\]\s*(?:(?:\[generate\]<Motion>(?:<Motion_\d{1,3}>)+</Motion>\[/generate\]\s*\[plan\].*?\[/plan\]\s*\[assess\].*?\[/assess\])\s*\[plan\].*?\[/plan\]\s*)*</think>\s*", **kwargs):
    """Check if the Qwen model output matches a specific format."""
    # import pdb; pdb.set_trace()
    # completion_contents = [completion[0]["content"] for completion in completions]
    completion_contents = [completion["text_content"] for completion in completions]

    matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
    # if False in matches:
    #     import pdb; pdb.set_trace()

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")
    return [1.0 if match else 0.0 for match in matches]
    
def format_reward_mogen_basic(completions, pattern=r"<think>.*?</think>\s*<answer>.*?<Motion>(?:<Motion_\d{1,3}>)+</Motion>.*?</answer>", **kwargs):
    """Check if the Qwen model output matches a specific format."""
    
    # pattern = r"<think>.*?</think>\s*<answer>.*?<Motion>.*?</Motion>.*?</answer>"
    # pattern = r"<think>.*?</think>\s*<answer>.*?<Motion>\s*(?=.*?(?:<Motion_\d{1,3}>\s*){2,})\s*</Motion>.*?</answer>"
    # pattern = r"<think>.*?</think>\s*<answer>.*?<Motion>\s*(?:<Motion_\d{1,3}>\s*){2,}.*?</Motion>.*?</answer>"
    
    
    
    # completion_contents = [completion[0]["content"] for completion in completions]
    completion_contents = [completion["text_content"] for completion in completions]

    matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
    # if False in matches:
    #     import pdb; pdb.set_trace()

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")
    return [1.0 if match else 0.0 for match in matches]


