import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from fire import Fire
from joblib import Parallel, delayed
from tqdm import tqdm

from marti.worlds.third_party.mcts_utils.ab_mcts.tasks.code.task import CodeProblem
from marti.worlds.third_party.mcts_utils.ab_mcts.tasks.math.task import MathProblem
from marti.worlds.third_party.mcts_utils.ab_mcts.tasks.base import Task


# sys.path.append("./experiments/arc2")
from marti.worlds.third_party.mcts_utils.utils import NodeState


def get_private_score(task: Task, node_state: NodeState | None) -> float:
    if node_state is not None:
        # print(node_state.generation_result)
        eval_results, _score = task.evaluate_on_test(
            llm_answer=node_state.generation_result
        )
        if len(eval_results) == 0:
            private_score = 0
        else:
            private_score = sum(
                [eval_result.get_score() for eval_result in eval_results]
            ) / len(eval_results)
    else:
        private_score = 0
    return private_score


def process_node(node, task):
    """Helper function to process a single node and calculate scores."""
    if node.expand_idx < 0:
        return None

    node_idx = node.expand_idx
    public_score = node.score

    # calc private score
    node_state = node.state
    # private_score = get_private_score(task, node_state)
    private_score = public_score


    return node_idx, public_score, private_score


def main(
    exp_name: str,
    n_jobs: int = 4,
    path_to_ckpt: str = "./outputs/ab-mcts/{exp_name}/{task_id}/checkpoints/checkpoint_n_answers_128.pkl",
    save_path: str = "./eval_outputs/ab-mcts/code/",
    path_to_code_tasks: str = "./data/arc_agi_2_eval_short.txt",
    force_rerun=True,
    task_type="code",
):
    with open(path_to_code_tasks, "r") as f:
        task_list = f.readlines()
    task_list = [t.strip() for t in task_list]
    # path_to_code_tasks is now a directory path
    subdirs = [d for d in os.listdir(path_to_code_tasks)
            if os.path.isdir(os.path.join(path_to_code_tasks, d))]

    # Index range
    task_indices = range(1, len(subdirs) + 1)


    node_idx_dict = {}
    public_scores_dict = {}
    private_scores_dict = {}
    for task_id in task_list:
        # TODO Replace with processed lcb data / load_file method also needs to change
        data_id = int(task_id) + 1

        if task_type == "math":
            math_problem_path = Path(f"./data/aime_problems_2025/2025-{data_id}.json")
            task = MathProblem.load_file(math_problem_path)
        elif task_type == "code":
            code_problem_path = Path(f"./data/json_outputs/output_{data_id}.json")
            task = CodeProblem.load_file(code_problem_path)

        # load state
        #
        state_path = Path(path_to_ckpt.format(task_id=task_id, exp_name=exp_name))  # Replace empty task_id
        proc_ret_path = Path(state_path.as_posix().replace(".pkl", "_proc_result.json"))
        if not state_path.exists():
            print(f"State path {state_path} does not exist")
            continue
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        if proc_ret_path.exists() and not force_rerun:
            print(f"Processing result {proc_ret_path} already exists")
            with open(proc_ret_path, "r") as f:
                proc_ret = json.load(f)
            node_idx_list = proc_ret["node_idx_list"]
            public_scores = proc_ret["public_scores"]
            private_scores = proc_ret["private_scores"]
        else:
            # Filter nodes with expand_idx >= 0 before processing
            valid_nodes = [
                node for node in state.tree.get_nodes() if node.expand_idx >= 0
            ]

            # Parallel processing of nodes
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(process_node)(node, task)
                for node in tqdm(valid_nodes, desc=f"Processing {task_id}")
            )

            # Filter out None results (from nodes with expand_idx < 0, though already filtered)
            results = [r for r in results if r is not None]

            # Sort results by node_idx (the first element of each tuple in results)
            # This ensures that node_idx_list, public_scores, and private_scores are ordered by node_idx
            results.sort(key=lambda x: x[0])

            # Unpack sorted results
            node_idx_list = []
            public_scores = []
            private_scores = []
            for result in results:
                node_idx, public_score, private_score = (
                    result  # No need to check for None here, already filtered
                )
                node_idx_list.append(node_idx)
                public_scores.append(public_score)
                private_scores.append(private_score)
            proc_ret = {
                "node_idx_list": node_idx_list,
                "public_scores": public_scores,
                "private_scores": private_scores,
            }
            with open(proc_ret_path, "w") as f:
                json.dump(proc_ret, f)

        node_idx_dict[task_id] = node_idx_list
        public_scores_dict[task_id] = public_scores
        private_scores_dict[task_id] = private_scores

    df_reward = pd.DataFrame(public_scores_dict)
    df_test = pd.DataFrame(private_scores_dict)

    df_reward.to_csv(
        os.path.join(save_path.format(exp_name=exp_name), "df_reward.csv"), index=False
    )
    df_test.to_csv(
        os.path.join(save_path.format(exp_name=exp_name), "df_test.csv"), index=False
    )

    print(f"Saved to {save_path.format(exp_name=exp_name)}")
    print(f"Quick results: {df_test.max(0).sum()} / {df_test.shape[1]}")
    print(f"Quick results: {df_reward.max(0).sum()} / {df_reward.shape[1]}")


if __name__ == "__main__":
    Fire(main)
# Problem, test cases, answers corresponding to test cases;