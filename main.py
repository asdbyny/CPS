import os
import json
from pathlib import Path
from src.utils import load_problems_from_subset
from src.generation import generate_solution
from src.evaluation import evaluate_single_sample

# 初始化路径
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在


def main():
    # 配置参数
    model_name = "gpt-4o"  # 可替换为其他模型
    k = 2  # 参考解数量（需在 configs 中配置）

    # 1. 加载问题数据
    print("🔍 加载问题数据...")
    problems = load_problems_from_subset()
    if not problems:
        print("❌ 未加载到任何问题，程序退出")
        return
    print(f"✅ 成功加载 {len(problems)} 个问题")

    # 2. 生成并评估解决方案
    print("\n🚀 开始生成与评估...")
    results = []
    for idx, problem in enumerate(problems, 1):
        try:
            # 生成解决方案
            solution = generate_solution(
                model_name=model_name,
                problem_text=problem["text"],
                reference_solutions=problem["reference_solutions"],
                k=k
            )

            # 评估解决方案
            eval_result = evaluate_single_sample(
                solution=solution["solution_text"],
                problem_text=problem["text"],
                difficulty_type=problem["difficulty_type"],
                reference_solutions=problem["reference_solutions"],
                k=k
            )

            # 补充元信息
            eval_result.update({
                "competition_id": problem["competition_id"],
                "problem_id": problem["problem_id"],
                "model_name": model_name,
                "sample_index": idx
            })
            results.append(eval_result)
            print(f"✅ 完成 {idx}/{len(problems)}：问题 {problem['problem_id']}，CPS={eval_result['cps']}")

        except Exception as e:
            print(f"❌ 处理问题 {problem.get('problem_id', idx)} 时出错：{str(e)}")
            continue

    # 3. 保存结果
    output_path = OUTPUT_DIR / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📊 评估完成，结果保存至：{output_path}")


if __name__ == "__main__":
    main()