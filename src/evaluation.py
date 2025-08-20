import json
import numpy as np
from typing import List, Dict
from pathlib import Path
from src.utils import is_rewrite

# 路径与配置加载（确保绝对路径）
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "ieir2025_config.json"

try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    CPS_PARAMS = config["cps_params"]
    DWE_PARAMS = config["dwe_params"]
except FileNotFoundError:
    print(f"❌ 错误：未找到 {CONFIG_PATH}，请检查 configs 目录")
    # 提供默认配置避免崩溃
    CPS_PARAMS = {"gamma": 1.0, "beta": 0.6, "k": [1, 2, 3, 4]}
    DWE_PARAMS = {"alpha_min": 0.3, "alpha_max": 0.8}
except KeyError as e:
    print(f"❌ 错误：配置文件缺少 {e} 字段，请检查格式")
    CPS_PARAMS = {"gamma": 1.0, "beta": 0.6, "k": [1, 2, 3, 4]}
    DWE_PARAMS = {"alpha_min": 0.3, "alpha_max": 0.8}

# 难度类型→难度值 d 映射（严格对应论文表1）
DIFFICULTY_TYPE_MAPPING = {
    "AMC_8": 0.00,
    "AMC_10": 0.20,
    "AMC_12": 0.30,
    "AHSME": 0.45,
    "AIME": 0.60,
    "USAJMO": 0.80,
    "USAMO": 0.90,
    "IMO": 1.00
}


def check_correctness(solution: str, problem_text: str) -> bool:
    """验证解答正确性（模拟3个LLM评审，实际需替换为API调用）"""
    # 临时逻辑：默认正确，实际使用时需调用 GPT-4o 等模型评审
    return True


def calculate_novelty_score(
        solution: str,
        reference_solutions: List[str],
        k: int = 1
) -> float:
    """计算新颖性得分（是否与参考解不同）"""
    if k not in CPS_PARAMS["k"]:
        print(f"⚠️ 警告：k 值 {k} 不在配置范围内，使用默认 k=1")
        k = 1

    # 检查是否与前 k 个参考解均不同
    is_distinct = all(
        not is_rewrite(solution, ref)
        for ref in reference_solutions[:k] if ref.strip()
    )
    return CPS_PARAMS["gamma"] * (1.0 if is_distinct else 0.0)


def is_step_valid(step: str) -> bool:
    """判断步骤是否有效（含数学逻辑关键词）"""
    valid_keywords = ["because", "since", "therefore", "=", "+", "-", "*", "/", "solve", "implies"]
    invalid_phrases = ["error", "wrong", "incorrect", "mistake"]
    step_lower = step.lower()
    has_valid = any(kw in step_lower for kw in valid_keywords)
    has_invalid = any(ph in step_lower for ph in invalid_phrases)
    return has_valid and not has_invalid


def is_step_redundant(step: str) -> bool:
    """判断步骤是否冗余（短文本或重复信息）"""
    step_lower = step.lower()
    return (len(step.strip()) < 10  # 过短步骤
            or "as before" in step_lower
            or "same as" in step_lower
            or "repeat" in step_lower)


def calculate_process_score(steps: List[str]) -> float:
    """计算过程质量得分（有效性+冗余性）"""
    if not steps:
        return 0.0

    # 有效步骤占比
    valid_steps = [step for step in steps if is_step_valid(step)]
    s_validity = len(valid_steps) / len(steps)

    # 冗余步骤占比
    redundant_steps = [step for step in steps if is_step_redundant(step)]
    s_redundancy = len(redundant_steps) / len(steps)

    # 合并得分（β=0.6）
    return round(
        CPS_PARAMS["beta"] * s_validity + (1 - CPS_PARAMS["beta"]) * (1 - s_redundancy),
        3
    )


def get_difficulty_score(difficulty_type: str) -> float:
    """获取问题难度值 d（基于竞赛类型）"""
    d = DIFFICULTY_TYPE_MAPPING.get(difficulty_type)
    if d is None:
        print(f"⚠️ 警告：未知难度类型 {difficulty_type}，使用默认 d=0.5")
        return 0.5
    return d


def calculate_alpha(d: float) -> float:
    """计算动态权重 α（基于难度 d）"""
    return round(
        DWE_PARAMS["alpha_min"] + (DWE_PARAMS["alpha_max"] - DWE_PARAMS["alpha_min"]) * d,
        3
    )


def calculate_cps(
        novelty_score: float,
        process_score: float,
        difficulty_type: str
) -> Dict:
    """计算创造性过程得分 CPS"""
    d = get_difficulty_score(difficulty_type)
    alpha = calculate_alpha(d)
    cps = round(alpha * novelty_score + (1 - alpha) * process_score, 3)
    return {"d": d, "alpha": alpha, "cps": cps}


def evaluate_single_sample(
        solution: str,
        problem_text: str,
        difficulty_type: str,
        reference_solutions: List[str],
        k: int = 1
) -> Dict:
    """完整评估单一样本"""
    # 1. 正确性检查（不正确则直接返回0分）
    correctness = check_correctness(solution, problem_text)
    if not correctness:
        return {
            "correctness": False,
            "novelty_score": 0.0,
            "process_score": 0.0,
            "difficulty_d": get_difficulty_score(difficulty_type),
            "alpha": calculate_alpha(get_difficulty_score(difficulty_type)),
            "cps": 0.0
        }

    # 2. 拆分步骤（按换行分割，跳过空行）
    steps = [step.strip() for step in solution.split("\n") if step.strip()]

    # 3. 计算各项得分
    novelty_score = calculate_novelty_score(solution, reference_solutions, k)
    process_score = calculate_process_score(steps)
    cps_result = calculate_cps(novelty_score, process_score, difficulty_type)

    return {
        "correctness": correctness,
        "novelty_score": novelty_score,
        "process_score": process_score,
        "difficulty_d": cps_result["d"],
        "alpha": cps_result["alpha"],
        "cps": cps_result["cps"]
    }