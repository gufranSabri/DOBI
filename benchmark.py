import argparse
import json
import os
from datetime import datetime

import lm_eval
from lm_eval.api.registry import get_model

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


# Loglikelihood-scored (forward-only, e.g. multiple choice) vs generate_until tasks
# (need .generate()). FlowModel supports both kinds:
#   loglikelihood -> forward() (K-step Euler integration once per batch);
#   generate_until -> autoregressive GenerationMixin.generate() (KV-cached).
LOGLIKELIHOOD_TASKS = ["mmlu", "mmlu_pro", "mathqa"]
GENERATIVE_TASKS = ["gsm8k", "humaneval", "mbpp"]

NUM_FEWSHOT = {
    "humaneval": 0,
    "mbpp": 0,
    "gsm8k": 0,
    "mmlu": 0,
    "mmlu_pro": 0,
    "mathqa": 0,
}
DEFAULT_FEWSHOT = 0

# HumanEval/MBPP's base task (doc_to_text = raw prompt, no gen_prefix) is written for
# BASE-style code continuation: the scoring filter (utils.build_predictions) glues the
# model's raw output directly onto the function signature and expects it to be more
# Python, nothing else. A chat-templated Instruct model instead replies conversationally
# ("To implement this, we need to...\n```python\n...") — that whole reply gets glued on
# as "code" and fails to parse, scoring ~0% regardless of whether the solution inside the
# fence was actually correct. lm-eval ships an "_instruct" variant of each (gen_prefix
# primes a direct code reply; build_predictions_instruct extracts the fenced block) —
# swap to it whenever the chat template is on, instead of chat-templating the base task.
INSTRUCT_TASK_VARIANT = {
    "humaneval": "humaneval_instruct",
    "mbpp": "mbpp_instruct",
}


def resolve_task_name(task: str, use_chat_template: bool) -> str:
    if use_chat_template and task in INSTRUCT_TASK_VARIANT:
        return INSTRUCT_TASK_VARIANT[task]
    return task


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


def resolve_use_chat_template(args) -> bool:
    """Each model should be evaluated the way it was tuned to be used. Qwen (and most
    HF orgs) name instruction-tuned checkpoints with "instruct" in the id — those were
    RLHF/SFT'd to expect the chat template (system/user turns) and underperform without
    it; base checkpoints have no chat template to apply at all. A DOBI checkpoint
    inherits its frozen student's behavior (utils/data.py trains it on chat-templated
    inputs most of the time — see RAW_TEXT_FRACTION), so the same "instruct" convention
    is checked against the saved config's base_model id, not the checkpoint PATH.
    --chat-template/--no-chat-template override this heuristic either way.
    """
    if args.chat_template is not None:
        return args.chat_template

    if args.model_type == "hf":
        return "instruct" in args.model.lower()

    # flow: args.model is a checkpoint DIRECTORY, not a model id — the
    # "instruct" signal lives in the saved config's base_model field instead.
    with open(os.path.join(args.model, "config.json")) as f:
        saved_config = json.load(f)
    base_model = saved_config.get("base_model", "")
    return "instruct" in base_model.lower()


def build_model(args):
    """Construct the lm-eval model wrapper for one of two targets:
      * --model-type hf  : a plain HF model/tokenizer id (baseline).
      * --model-type flow: a DOBI flow-matching checkpoint dir (registered as
        'dobi-flow' in lm_eval/models/dobi.py).
    """
    if args.model_type == "hf":
        model_cls = get_model("hf")
        return model_cls(pretrained=args.model, max_length=args.max_length, truncation=True)

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

    use_chat_template = resolve_use_chat_template(args)
    print(f"apply_chat_template={use_chat_template} "
          f"({'explicit override' if args.chat_template is not None else 'auto-detected from model id'})")

    tasks = LOGLIKELIHOOD_TASKS + GENERATIVE_TASKS if not args.tasks else args.tasks
    all_results = {}

    for task in tasks:
        eval_task = resolve_task_name(task, use_chat_template)
        print(f"\n{'='*60}\nEvaluating: {task}"
              + (f"  (using {eval_task})" if eval_task != task else "")
              + f"\n{'='*60}")
        num_fewshot = NUM_FEWSHOT.get(task, DEFAULT_FEWSHOT)
        print(f"  num_fewshot={num_fewshot}")
        if args.limit is not None:
            print(f"  limit={args.limit} (testing on a subset only)")

        results = lm_eval.simple_evaluate(
            model=model,
            tasks=[eval_task],
            num_fewshot=num_fewshot,
            confirm_run_unsafe_code=True,
            log_samples=True,
            limit=args.limit,
            apply_chat_template=use_chat_template,
            fewshot_as_multiturn=use_chat_template,
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
        f.write(f"Model type       : {args.model_type}\n")
        f.write(f"Model            : {args.model}\n")
        f.write(f"Chat template    : {use_chat_template}\n")
        f.write(f"Date             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for task, task_results in all_results.items():
            f.write(format_results(task, {"results": {task: task_results.get(task, task_results)}}))
    print(f"\nAll done. Summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, help="Directory to store benchmark results")
    parser.add_argument("--model-type", choices=["hf", "flow"], required=True,
                        help="'hf' evaluates a plain base HF model; 'flow' evaluates a DOBI flow checkpoint dir.")
    parser.add_argument("--model", required=True,
                         help="HF hub id (--model-type hf) or a checkpoint dir written by save_hf_model (flow).")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Override the default task list (mmlu, mmlu_pro, mathqa, gsm8k, humaneval, mbpp).")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Override max sequence length.")
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit examples per task (int count, or <1 for a fraction). For quick testing.")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Ablation override for the flow Euler-step count. Default: the value trained/saved "
                             "in the checkpoint config.")
    chat_group = parser.add_mutually_exclusive_group()
    chat_group.add_argument("--chat-template", dest="chat_template", action="store_true", default=None,
                             help="Force-wrap prompts in the model's chat template. Default: auto-detected from "
                                  "'instruct' in the model id (or the DOBI checkpoint's saved base_model id).")
    chat_group.add_argument("--no-chat-template", dest="chat_template", action="store_false",
                             help="Force raw completion prompts, no chat template, even for an Instruct model.")
    args = parser.parse_args()

    main(args)
