import argparse
import json
import os
from datetime import datetime

import lm_eval
from lm_eval.api.registry import get_model

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


# Loglikelihood-scored (forward-only, e.g. multiple choice) vs generate_until tasks
# (need .generate()). Both DOBI model types support both kinds:
#   * DiffusionModel: loglikelihood -> forward(cont_spans=...) single masked pass;
#     generate_until -> confidence-unmasking generate().
#   * FlowModel: loglikelihood -> forward() (K-step Euler integration once per batch);
#     generate_until -> autoregressive GenerationMixin.generate() (KV-cached).
LOGLIKELIHOOD_TASKS = ["mmlu"]
GENERATIVE_TASKS = ["gsm8k", "humaneval"]

NUM_FEWSHOT = {
    "humaneval": 0,
    "gsm8k": 0,
    "mmlu": 0,
}
DEFAULT_FEWSHOT = 0


def format_results(task: str, results: dict) -> str:
    lines = [
        "=" * 60,
        f"Task: {task}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ]
    task_results = results.get("results", {}).get(task, {})
    if not task_results:
        task_results = next(iter(results.get("results", {}).values()), {})
    for metric, value in task_results.items():
        if isinstance(value, float):
            lines.append(f"  {metric}: {value:.4f}")
        else:
            lines.append(f"  {metric}: {value}")
    lines.append("")
    return "\n".join(lines)


def build_model(args):
    """Construct the lm-eval model wrapper for one of three targets:
      * --model-type hf       : a plain HF model/tokenizer id (baseline).
      * --model-type diffusion: a DOBI diffusion checkpoint dir (registered as
        'dobi-diffusion' in lm_eval/models/dobi.py).
      * --model-type flow     : a DOBI flow-matching checkpoint dir (registered as
        'dobi-flow').
    """
    if args.model_type == "hf":
        model_cls = get_model("hf")
        return model_cls(pretrained=args.model, max_length=args.max_length, truncation=True)

    if args.model_type == "diffusion":
        model_cls = get_model("dobi-diffusion")
        kwargs = {"max_length": args.max_length} if args.max_length else {}
        if args.num_steps is not None:
            kwargs["num_sampling_steps"] = args.num_steps
        return model_cls(pretrained=args.model, **kwargs)

    if args.model_type == "flow":
        model_cls = get_model("dobi-flow")
        kwargs = {"max_length": args.max_length} if args.max_length else {}
        if args.num_steps is not None:
            kwargs["num_flow_steps"] = args.num_steps
        return model_cls(pretrained=args.model, **kwargs)

    raise ValueError(f"Unknown --model-type '{args.model_type}'")


def main(args):
    benchmark_dir = os.path.join(args.work_dir, "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)

    print(f"Loading {args.model_type} model from {args.model} …")
    model = build_model(args)

    tasks = LOGLIKELIHOOD_TASKS + GENERATIVE_TASKS if not args.tasks else args.tasks
    all_results = {}

    for task in tasks:
        print(f"\n{'='*60}\nEvaluating: {task}\n{'='*60}")
        num_fewshot = NUM_FEWSHOT.get(task, DEFAULT_FEWSHOT)
        print(f"  num_fewshot={num_fewshot}")
        if args.limit is not None:
            print(f"  limit={args.limit} (testing on a subset only)")

        results = lm_eval.simple_evaluate(
            model=model,
            tasks=[task],
            num_fewshot=num_fewshot,
            confirm_run_unsafe_code=True,
            log_samples=True,
            limit=args.limit,
        )

        if results is None:  # non-main rank under data-parallel
            continue

        task_results = results.get("results", {})
        all_results[task] = task_results

        task_dir = os.path.join(benchmark_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        for sample_task, sample_list in results.get("samples", {}).items():
            samples_path = os.path.join(task_dir, f"samples_{sample_task}.jsonl")
            with open(samples_path, "w") as f:
                for sample in sample_list:
                    f.write(json.dumps(sample, default=str) + "\n")
            print(f"  Saved {len(sample_list)} samples → {samples_path}")

        with open(os.path.join(task_dir, "results.txt"), "w") as f:
            f.write(format_results(task, results))
        with open(os.path.join(task_dir, "results.json"), "w") as f:
            json.dump(task_results, f, indent=2)
        print(f"  Saved → {task_dir}/results.{{txt,json}}")

    if not all_results:
        return

    summary_path = os.path.join(benchmark_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Evaluation Summary\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Model     : {args.model}\n")
        f.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for task, task_results in all_results.items():
            f.write(format_results(task, {"results": {task: task_results.get(task, task_results)}}))
    print(f"\nAll done. Summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, help="Directory to store benchmark results")
    parser.add_argument("--model-type", choices=["hf", "diffusion", "flow"], required=True,
                        help="'hf' evaluates a plain base HF model; 'diffusion'/'flow' evaluate a DOBI checkpoint dir.")
    parser.add_argument("--model", required=True,
                         help="HF hub id (--model-type hf) or a checkpoint dir written by save_hf_model (diffusion/flow).")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Override the default task list (mmlu + gsm8k + humaneval).")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Override max sequence length. Diffusion checkpoints are additionally capped to the "
                             "trained MaskUNet's max_seq_len - 1 regardless of this flag.")
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit examples per task (int count, or <1 for a fraction). For quick testing.")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Ablation override for the diffusion sampling-step count K, or the flow Euler-step "
                             "count. Default: the value trained/saved in the checkpoint config.")
    args = parser.parse_args()

    main(args)
