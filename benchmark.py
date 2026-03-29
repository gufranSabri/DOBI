import argparse
import json
import os
from datetime import datetime

import lm_eval
from lm_eval.models.huggingface import HFLM


TASKS = [
    "mmlu",
    # "mmlu_pro",
    # "gsm8k",
    # "mathqa",
    # "humaneval",
    # "mbpp",
]

def format_results(task: str, results: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"Task: {task}")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    task_results = results.get("results", {}).get(task, {})
    if not task_results:
        # some tasks return results under a slightly different key
        task_results = next(iter(results.get("results", {}).values()), {})

    for metric, value in task_results.items():
        if isinstance(value, float):
            lines.append(f"  {metric}: {value:.4f}")
        else:
            lines.append(f"  {metric}: {value}")

    lines.append("")
    return "\n".join(lines)


def main(args):
    os.makedirs(args.work_dir, exist_ok=True)

    print(f"Loading model from {args.excited_model_path} …")
    model = HFLM(
        pretrained="Qwen/Qwen3.5-0.8B",
        excited_model_path=args.excited_model_path,
    )

    all_results = {}

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {task}")
        print(f"{'='*60}")

        try:
            results = lm_eval.simple_evaluate(
                model=model,
                tasks=[task],
                num_fewshot=0,
            )

            task_results = results.get("results", {})
            all_results[task] = task_results

            # --- per-task txt file ---
            txt_path = os.path.join(args.work_dir, f"{task}_results.txt")
            with open(txt_path, "w") as f:
                f.write(format_results(task, results))
            print(f"  Saved → {txt_path}")

            # --- per-task json file (full, lossless) ---
            json_path = os.path.join(args.work_dir, f"{task}_results.json")
            with open(json_path, "w") as f:
                json.dump(task_results, f, indent=2)
            print(f"  Saved → {json_path}")

        except Exception as e:
            print(f"  ERROR on task '{task}': {e}")
            all_results[task] = {"error": str(e)}

            err_path = os.path.join(args.work_dir, f"{task}_results.txt")
            with open(err_path, "w") as f:
                f.write(f"Task: {task}\nERROR: {e}\n")

    # --- combined summary ---
    summary_path = os.path.join(args.work_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"Model : Qwen/Qwen3.5-0.8B\n")
        f.write(f"Aligned: {args.excited_model_path}\n")
        f.write(f"Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for task, task_results in all_results.items():
            f.write(format_results(task, {"results": {task: task_results.get(task, task_results)}}))

    print(f"\nAll done. Summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir",           default="work_dir/test")
    parser.add_argument("--excited-model-path", dest="excited_model_path", default="work_dir/50k")
    args = parser.parse_args()

    main(args)
