from typing import List, Dict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def build_prompt(
        problem_text: str,
        reference_solutions: List[str],
        k: int = 1
) -> str:
    """构建提示词（适配论文要求的创造性+严谨性）"""
    # 读取提示词模板（不存在则用默认模板）
    prompt_template = """Solve the following mathematical problem with a creative and logically rigorous solution.
Problem: {problem}
Reference solutions (avoid repetition, use different methods):
{reference_solutions}
Requirements:
1. The solution must be distinct from the references (not simple rephrasing).
2. Steps must be clear, error-free, and non-redundant.
3. The final answer must be correct.
Solution:"""

    # 填充参考解
    ref_text = "\n".join([
        f"Reference {i + 1}: {ref[:100]}..."  # 截断过长的参考解
        for i, ref in enumerate(reference_solutions[:k]) if ref.strip()
    ])
    if not ref_text:
        ref_text = "No reference solutions provided."

    return prompt_template.format(
        problem=problem_text,
        reference_solutions=ref_text
    )


def generate_solution(
        model_name: str,
        problem_text: str,
        reference_solutions: List[str],
        k: int = 1
) -> Dict:
    """生成解决方案（模拟模型输出，实际需替换为API调用）"""
    prompt = build_prompt(problem_text, reference_solutions, k)

    # 模拟模型输出（实际使用时替换为 OpenAI/Gemini 等API调用）
    simulated_solution = f"To solve the problem: {problem_text[:50]}...\n1. First step: Analyze the problem.\n2. Second step: Apply mathematical operations.\n3. Conclusion: The result is obtained."

    return {
        "solution_text": simulated_solution,
        "reference_solutions_used": reference_solutions[:k],
        "k": k,
        "prompt_used": prompt  # 保存提示词用于调试
    }