import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
from marti.agents.base_agent import Agent
from marti.agents.math_agent import MathSolver
import ray
import json
from copy import deepcopy
import random
from vllm import SamplingParams

def get_kwargs(
    args,
):
    spatial_adj_mats = []
    temporal_adj_mats = []
    agent_ids = [list(agent.keys())[0] for agent in args.agents]
    agent_roles = [list(agent.values())[0]['role'] for agent in args.agents]
    N = len(agent_roles)

    if args.agent_workflow == 'chain-of-agents':
        for turn_id in range(args.workflow_args.num_rounds):
            spatial_adj_mat = [[1 if i == j and i == turn_id else 0 for j in range(N)] for i in range(N)]
            temporal_adj_mat = [[0 for j in range(N)] for i in range(N)]
            if turn_id == 1:  # verifier
                temporal_adj_mat[0][1] = 1
            elif turn_id == 2:  # refiner
                temporal_adj_mat[0][2] = 1
                temporal_adj_mat[1][2] = 1
            spatial_adj_mats.append(spatial_adj_mat)
            temporal_adj_mats.append(temporal_adj_mat)
    elif args.agent_workflow == 'mixture-of-agents':
        for turn_id in range(args.workflow_args.num_rounds):
            if turn_id < args.workflow_args.num_rounds - 1:  # generator
                spatial_adj_mat = [[1 if i == j and i != N - 1 else 0 for j in range(N)] for i in range(N)]
                temporal_adj_mat = [[0 for j in range(N)] for i in range(N)]
            else:  # aggregator
                spatial_adj_mat = [[1 if i == j and i == N - 1 else 0 for j in range(N)] for i in range(N)]
                temporal_adj_mat = [[1 if j == N - 1 and i != N - 1 else 0 for j in range(N)] for i in range(N)]
            spatial_adj_mats.append(spatial_adj_mat)
            temporal_adj_mats.append(temporal_adj_mat)
    elif args.agent_workflow == 'multi-agents-debate':
        for turn_id in range(args.workflow_args.num_rounds):
            spatial_adj_mat = [[1 if i == j else 0 for j in range(N)] for i in range(N)]
            temporal_adj_mat = [[1 if i != j else 0 for j in range(N)] for i in range(N)]
            if args.workflow_args.contain_self:
                for i in range(N):
                    temporal_adj_mat[i][i] = 1
            spatial_adj_mats.append(spatial_adj_mat)
            temporal_adj_mats.append(temporal_adj_mat)
    elif args.agent_workflow == 'majority-voting':
        for turn_id in range(args.workflow_args.num_rounds):
            spatial_adj_mat = [[1 if i == j else 0 for j in range(N)] for i in range(N)]
            temporal_adj_mat = [[0 for j in range(N)] for i in range(N)]
            spatial_adj_mats.append(spatial_adj_mat)
            temporal_adj_mats.append(temporal_adj_mat)
    else:
        raise NotImplementedError(f"The workflow {args.agent_workflow} is not implemented yet")

    prompt = {agent_role: {} for agent_role in agent_roles}
    for i, (agent_id, agent_role) in enumerate(zip(agent_ids, agent_roles)):
        if args.agents[i].get(agent_id, {}).get('system_prompt', []):
            prompt[agent_role]['system_prompt'] = args.agents[i][agent_id]['system_prompt'][0]
        for j in range(args.workflow_args.num_rounds):
            custom_prompt = args.agents[i].get(agent_id, {}).get('chat_template', [])
            try:
                if custom_prompt:
                    if len(custom_prompt) < args.workflow_args.num_rounds:
                        prompt[agent_role][str(j)] = custom_prompt[0]
                    else:
                        prompt[agent_role][str(j)] = custom_prompt[j]
                elif hasattr(args, 'chat_template') and args.chat_template:
                    if len(args.chat_template) < args.workflow_args.num_rounds:
                        prompt[agent_role][str(j)] = args.chat_template[0]
                    else:
                        prompt[agent_role][str(j)] = args.chat_template[j]
            except Exception as e:
                raise ValueError(f"Prompt for agent {agent_id} ({agent_role}) is not defined for round {j}")

    n = getattr(args, "n_samples_per_prompt", 1)
    temperature = getattr(args, "temperature", 0.6)
    top_p = getattr(args, "top_p", 1.0)
    top_k = getattr(args, "top_k", -1)
    max_tokens = getattr(args, "generate_max_len", 512)
    min_tokens = getattr(args, "min_new_tokens", 1)
    skip_special_tokens = getattr(args, "skip_special_tokens", False)

    # Optional sampling strategy knob; if set to "demo", use a multi-sample, moderately random setup
    sampling_strategy = getattr(args, "sampling_strategy", None)
    if sampling_strategy == "demo":
        n = max(3, int(n) if isinstance(n, int) else 1)
        temperature = max(0.6, float(temperature))
        top_p = 0.95 if top_p is None else min(0.95, float(top_p))
        top_k = -1 if top_k is None else int(top_k)

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        skip_special_tokens=skip_special_tokens,
    )

    return {
        "prompt": prompt,
        "agent_ids": agent_ids,
        "agent_roles": agent_roles,
        "spatial_adj_mats": spatial_adj_mats,
        "temporal_adj_mats": temporal_adj_mats,
    "sampling_params": sampling_params,
    }


class MAGraph(ABC):
    """
    Multi-agent graph.
    """

    def __init__(
        self,
        agents: List[Dict],
        agent_ids: List[str],
        agent_roles: List[str],
        agent_workflow: str,
        prompt: Dict,
        spatial_adj_mats: List[List[List[int]]],
        temporal_adj_mats: List[List[List[int]]],
        sampling_params,
        *args, **kwargs
    ):
        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.agents = agents
        # Map from node/agent id string to index in self.agents
        # The previous implementation expected a non-existent `agent_id` key in each agent dict.
        # We instead rely on the provided `agent_ids` list, which is used to initialize MathSolver ids.
        self.id2agent = {agent_ids[i]: i for i in range(len(agents))}
        self.agent_ids: List[str] = agent_ids
        self.agent_roles: List[str] = agent_roles
        self.agent_workflow = agent_workflow
        self.prompt = prompt
        self.nodes: Dict[str, Agent] = {}
        self.id2num = {}
        self.potential_spatial_edges: List[List[str, str]] = []
        self.potential_temporal_edges: List[List[str, str]] = []
        self.sampling_params = sampling_params
        self.shuffle_responses = kwargs.get('shuffle_responses', False)
        self.contain_self = kwargs.get('contain_self', False)

        self.history = []  # Store the historical records of each round

        self.init_nodes()  # add nodes to the self.nodes
        self.init_potential_edges()  # add potential edges to the self.potential_spatial/temporal_edges

        self.spatial_adj_mats = torch.tensor(spatial_adj_mats).view(len(spatial_adj_mats), -1)
        self.temporal_adj_mats = torch.tensor(temporal_adj_mats).view(len(temporal_adj_mats), -1)

    def spatial_adj_matrix(self):
        return np.array([
            [1 if self.nodes[id2] in self.nodes[id1].spatial_successors else 0 for id2 in self.nodes] for id1 in self.nodes
        ])

    def temporal_adj_matrix(self):
        return np.array([
            [1 if self.nodes[id2] in self.nodes[id1].temporal_successors else 0 for id2 in self.nodes] for id1 in self.nodes
        ])

    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges

    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among {[node.id for node in self.nodes.values()]}")

    def add_node(self, node: Agent):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        self.id2num[node_id] = len(self.nodes) - 1
        return node

    def init_nodes(self):
        for agent, agent_id, agent_role in zip(self.agents, self.agent_ids, self.agent_roles):
            agent_instance = MathSolver(
                id=agent_id,
                pretrain=agent["pretrain"],
                role=agent_role,
                agent_workflow=self.agent_workflow,
                prompt=self.prompt,
                is_reasoning_model=agent["is_reasoning_model"],
                shuffle_responses=self.shuffle_responses
            )
            self.add_node(agent_instance)

    def init_potential_edges(self):
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id, node2_id])
                self.potential_temporal_edges.append([node1_id, node2_id])

    def clear_spatial_connection(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []

    def clear_temporal_connection(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            if node_id != self.decision_node.id:
                self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, turn_id):
        self.clear_spatial_connection()

        for potential_connection, edge_mask in zip(self.potential_spatial_edges, self.spatial_adj_mats[turn_id]):
            out_node: Agent = self.find_node(potential_connection[0])
            in_node: Agent = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'spatial')
                continue

    def construct_temporal_connection(self, turn_id):
        self.clear_temporal_connection()

        if turn_id > 0:
            for potential_connection, edge_mask in zip(self.potential_temporal_edges, self.temporal_adj_mats[turn_id]):
                out_node: Agent = self.find_node(potential_connection[0])
                in_node: Agent = self.find_node(potential_connection[1])
                if edge_mask == 0.0:
                    continue
                elif edge_mask == 1.0:
                    if not self.check_cycle(in_node, {out_node}):
                        out_node.add_successor(in_node, 'temporal')
                    continue

    def should_run(self, node_id, turn_id):
        return self.spatial_adj_mats[turn_id][self.id2num[node_id] * int(self.spatial_adj_mats.shape[-1] ** 0.5) + self.id2num[node_id]] == 1

    def run_node(self, refs, node_infos, node_id, inputs, turn_id):
        messages = self.nodes[node_id].execute(inputs, turn_id, self.contain_self)  # output is saved in the node.outputs
        agent = self.agents[self.id2agent[node_id]]
        prompts = []
        for message in messages:
            prompt = agent["tokenizer"].apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            if agent.get("is_reasoning_model", False) and agent.get("enable_thinking", agent.get("is_reasoning_model", False)):
                prompt = prompt + "<think>"
            elif "qwen3" in agent.get("pretrain", "").lower():
                prompt = prompt + "<think>\n\n</think>\n\n"
            prompts.append(prompt)

        sampling_params = deepcopy(self.sampling_params)
        budget = sampling_params.n
        if sampling_params.n > 1:
            prompts = prompts * sampling_params.n
            sampling_params.n = 1

        batch_size = (len(prompts) + len(agent["llms"]) - 1) // len(agent["llms"])
        for i, llm in enumerate(agent["llms"]):
            batch_prompts = prompts[i * batch_size: (i + 1) * batch_size]
            batch_inputs = inputs[i * batch_size: (i + 1) * batch_size]
            batch_messages = messages[i * batch_size: (i + 1) * batch_size]
            if batch_prompts:
                refs.append(llm.generate.remote(batch_prompts, sampling_params=sampling_params))
                node_infos.append((node_id, batch_inputs, batch_messages))
        return refs, node_infos

    def post_process(self, node_id, result, messages):
        agent = self.agents[self.id2agent[node_id]]
        budget = self.sampling_params.n

        outputs = [output.outputs[0].text for output in result]
        code_outputs = []
        if budget > 1:
            arr = np.array(outputs)
            reshaped = arr.reshape(budget, -1)
            transposed = reshaped.T
            outputs = transposed.tolist()
            for output in outputs:
                random.shuffle(output)

        if agent.get("code_execution", False):  # extract Python code from the output
            code_outputs = [output.split("```python")[-1].split("```")[0].strip() for output in outputs]
            code_outputs = self.nodes[node_id].execute_code(code_outputs)
        if agent.get("is_reasoning_model", False) and agent.get("enable_thinking", agent.get("is_reasoning_model", False)):
            for i in range(len(outputs)):
                if len(messages) > i and len(messages[i]) > 0:
                    messages[i][-1]["content"] = messages[i][-1]["content"].rstrip("<think>")
                if isinstance(outputs[i], list):
                    for j in range(len(outputs[i])):
                        if not outputs[i][j].startswith("<think>"):
                            outputs[i][j] = "<think>" + outputs[i][j]
                elif isinstance(outputs[i], str):
                    if not outputs[i].startswith("<think>"):
                        outputs[i] = "<think>" + outputs[i]

        return messages, outputs, code_outputs

    def run(
        self,
        inputs: Any,
        num_rounds: int = 1,
        *args, **kwargs
    ) -> List[Any]:
        self.all_prompts = inputs
        history = [[] for _ in range(len(self.agents))]

        for turn_id in range(num_rounds):
            refs, node_infos = [], []
            self.construct_spatial_connection(turn_id)
            self.construct_temporal_connection(turn_id)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:  # Traverse the entire graph
                curr_node_id = zero_in_degree_queue.pop(0)
                if not self.should_run(curr_node_id, turn_id):
                    continue

                refs, node_infos = self.run_node(refs, node_infos, curr_node_id, inputs, turn_id)
                for successor in self.nodes[curr_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            results = ray.get(refs)
            for (node_id, batch_inputs, batch_messages), batch_results in zip(node_infos, results):
                batch_messages, batch_outputs, batch_code_outputs = self.post_process(node_id, batch_results, batch_messages)
                self.nodes[node_id].update_round_history(batch_inputs, batch_messages, batch_outputs, batch_code_outputs, turn_id)

            for node_id in self.nodes.keys():
                if self.should_run(node_id, turn_id):
                    node_history = self.nodes[node_id].history[turn_id]
                    history[self.id2agent[node_id]].append(node_history)

        self.history = history
        return history

    def get_history(self):
        # TODO: transform graph history into training format
        """
        chain: [
            [agent1, agent2, agent3] # prob 1
            [agent1, agent2, agent3] # prob 2
        ]
        mad: [
            [
                [turn11, turn12, turn13] # agent 1
                [turn21, turn22, turn23] # agent 2
            ] # prob 1
            [
                [turn11, turn12, turn13] # agent 1
                [turn21, turn22, turn23] # agent 2
            ] # prob 2
        ]
        mix: [
            [agent1(gen), agent2(gen), agent3(mix)] # prob 1
            [agent1(gen), agent2(gen), agent3(mix)] # prob 2
        ]
        """
        history_per_problem = []
        # history_per_problem = {problem: [] for problem in all_prompts}
        # problem2idx = {problem: self.all_prompts.index(problem) for problem in self.all_prompts}
        for problem_idx, problem in enumerate(self.all_prompts):
            # problem_idx = problem2idx[problem]
            problem_history = []
            for node_id, node_history in enumerate(self.history):
                agent_history = []
                for turn_id, round_history in enumerate(node_history):
                    inputs = round_history['inputs']
                    outputs = round_history['outputs']

                    agent_history.append({
                        'agent_id': round_history['agent_id'],
                        'agent_role': round_history['agent_role'],
                        'pretrain': round_history['pretrain'],
                        'turn_id': round_history['turn_id'],
                        'user': inputs[problem_idx],
                        'assistant': outputs[problem_idx],
                        'spatial_predecessors': round_history['spatial_predecessors'],
                        'temporal_predecessors': round_history['temporal_predecessors'],
                    })
                if len(agent_history) == 1:
                    agent_history = agent_history[0]
                # history_per_problem[problem].append(agent_history)
                problem_history.append(agent_history)
            history_per_problem.append(problem_history)
        return history_per_problem

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def save_history(self, history):
        with open("./history.json", "w") as f:
            json.dump(history, f, indent=4)
