import json
from typing import List, Dict
from pathlib import Path

# 项目根目录（自动适配不同运行环境）
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_problems_from_subset() -> List[Dict]:
    """从 data/subset.json 加载问题数据，适配实际字段结构"""
    subset_path = BASE_DIR / "data" / "subset.json"
    try:
        with open(subset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 解析并标准化问题数据（严格匹配 subset.json 字段）
        problems = []
        for item in data:
            problems.append({
                "text": item.get("problem", ""),  # 题目文本
                "difficulty_type": item.get("competition", "AMC_8"),  # 竞赛类型（映射难度）
                "reference_solutions": list(item.get("solutions", {}).values()),  # 参考解列表
                "competition_id": item.get("competition_id", ""),  # 原始竞赛ID
                "problem_id": item.get("problem_id", "")  # 原始题目ID
            })
        return problems

    except FileNotFoundError:
        print(f"❌ 错误：未找到 {subset_path}，请检查 data 目录下是否存在 subset.json")
        return []
    except json.JSONDecodeError:
        print(f"❌ 错误：{subset_path} 格式错误，请检查JSON语法")
        return []


def is_rewrite(solution: str, ref_solution: str) -> bool:
    """判断生成解是否为参考解的简单改写（防冗余逻辑）"""
    solution_lower = solution.lower().strip()
    ref_lower = ref_solution.lower().strip()

    # 长度差异过大则不是改写
    len_sol = len(solution_lower)
    len_ref = len(ref_lower)
    if len_sol == 0 or len_ref == 0:
        return False
    len_diff_ratio = abs(len_sol - len_ref) / max(len_sol, len_ref)
    if len_diff_ratio > 0.3:
        return False

    # 关键词重复率过高则判定为改写
    sol_words = set(solution_lower.split())
    ref_words = set(ref_lower.split())
    common_ratio = len(sol_words & ref_words) / max(len(sol_words), len(ref_words))
    return common_ratio > 0.7  # 阈值可根据实际调整