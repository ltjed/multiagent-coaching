import asyncio
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict
from typing import Type

from copy import deepcopy
import shortuuid
from class_registry import ClassRegistry


class Agent(ABC):
    """
    Agent operations.
    """

    def __init__(
        self,
        id: Optional[str],
        role: str = "",
        pretrain: str = "",
        agent_workflow: str = "",
        is_reasoning_model: bool = False,
    ):
        """
        Initializes a new Agent instance.
        """
        self.id: str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.role: str = role
        self.pretrain: str = pretrain
        self.agent_workflow: str = agent_workflow
        self.is_reasoning_model: bool = is_reasoning_model

        # communication & topology
        self.spatial_predecessors: List[Agent] = []
        self.spatial_successors: List[Agent] = []
        self.temporal_predecessors: List[Agent] = []
        self.temporal_successors: List[Agent] = []

        # history
        self.history: Dict[str, List[Any]] = {}

    def add_predecessor(self, operation: 'Agent', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Agent', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Agent', st='spatial'):
        if st == 'spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st == 'temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Agent', st='spatial'):
        if st == 'spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st == 'temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Agent] = []
        self.spatial_successors: List[Agent] = []
        self.temporal_predecessors: List[Agent] = []
        self.temporal_successors: List[Agent] = []

    def update_round_history(self, raw_inputs, inputs, outputs, code_outputs, turn_id: int):
        inputs = [item[-1]['content'] for item in inputs]
        if turn_id in self.history.keys():
            self.history[turn_id]['raw_inputs'].extend(deepcopy(raw_inputs))
            self.history[turn_id]['inputs'].extend(deepcopy(inputs))
            self.history[turn_id]['outputs'].extend(deepcopy(outputs))
            if code_outputs:
                self.history[turn_id]['code_outputs'].extend(deepcopy(code_outputs))
        else:
            round_history = {
                'agent_id': self.id,
                'agent_role': self.role,
                'pretrain': self.pretrain,
                'turn_id': turn_id,
                'raw_inputs': raw_inputs,
                'inputs': inputs,
                'outputs': outputs,
                'spatial_predecessors': [{node.id: node.role} for node in self.spatial_predecessors],
                'temporal_predecessors': [{node.id: node.role} for node in self.temporal_predecessors],
            }
            if code_outputs:
                round_history['code_outputs'] = code_outputs
            self.history[turn_id] = deepcopy(round_history)

    def get_spatial_info(self, turn_id: int) -> Dict[str, Dict]:
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                predecessor_outputs = predecessor.history.get(turn_id, {}).get('outputs', [])
                predecessor_code_outputs = predecessor.history.get(turn_id, {}).get('code_outputs', [])
                processed_outputs = []
                if predecessor_outputs:
                    for i in range(len(predecessor_outputs)):
                        output = predecessor_outputs[i]
                        if predecessor.is_reasoning_model:  # extract the reasoning model output
                            output = output.split("</think>")[-1].strip()
                        if predecessor_code_outputs:
                            output = f'{output}\n{predecessor_code_outputs[i]}'
                        processed_outputs.append(output)
                else:
                    print(f"Warning: No outputs found for the spatial predecessor ({predecessor.id}) of agent {self.id} in round {turn_id}.")
                spatial_info[predecessor.id] = {"role": predecessor.role, "output": processed_outputs}

        return spatial_info

    def get_temporal_info(self, turn_id: int, contain_self: bool) -> Dict[str, Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_outputs = []
                for r_id in range(turn_id - 1, -1, -1):  # find the latest output of the temporal predecessor
                    predecessor_outputs = predecessor.history.get(r_id, {}).get('outputs', [])
                    predecessor_code_outputs = predecessor.history.get(r_id, {}).get('code_outputs', [])
                    if predecessor_outputs:
                        break
                processed_outputs = []
                if predecessor_outputs:
                    for i in range(len(predecessor_outputs)):
                        output = predecessor_outputs[i]
                        if predecessor.is_reasoning_model:  # extract the reasoning model output
                            output = output.split("</think>")[-1].strip()
                        if predecessor_code_outputs:
                            output = f'{output}\n{predecessor_code_outputs[i]}'
                        processed_outputs.append(output)
                else:
                    print(f"Warning: No outputs found for the temporal predecessor ({predecessor.id}) of agent {self.id} in round {turn_id}.")
                temporal_info[predecessor.id] = {"role": predecessor.role, "output": processed_outputs}
        if contain_self:
            self_outputs = []
            for r_id in range(turn_id - 1, -1, -1):  # find the latest output of the temporal predecessor
                self_outputs = self.history.get(r_id, {}).get('outputs', [])
                self_code_outputs = self.history.get(r_id, {}).get('code_outputs', [])
                if self_outputs:
                    break
            processed_outputs = []
            if self_outputs:
                for i in range(len(self_outputs)):
                    output = self_outputs[i]
                    if self.is_reasoning_model:
                        output = output.split("</think>")[-1].strip()
                    if self_code_outputs:
                        output = f'{output}\n{self_code_outputs[i]}'
                    processed_outputs.append(output)
            else:
                print(f"Warning: No outputs found for the agent ({self.id}) in round {turn_id}.")
            temporal_info[self.id] = {"role": self.role, "output": processed_outputs}

        return temporal_info

    def execute(self, inputs: Any, turn_id: int, contain_self: bool, **kwargs):
        spatial_info: Dict[str, Dict] = self.get_spatial_info(turn_id)
        temporal_info: Dict[str, Dict] = self.get_temporal_info(turn_id, contain_self)
        results = self._execute(deepcopy(inputs), spatial_info, temporal_info, turn_id, **kwargs)
        return results

    @abstractmethod
    def _execute(self, inputs: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        pass

    @abstractmethod
    def _process_inputs(self, raw_inputs: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs) -> List[Any]:
        pass


class AgentRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, name: str, *args, **kwargs) -> Agent:
        return cls.registry.get(name, *args, **kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type:
        return cls.registry.get_class(name)