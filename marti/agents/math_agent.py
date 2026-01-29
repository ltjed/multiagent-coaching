from typing import List, Any, Dict, NamedTuple, Tuple

from marti.agents.base_agent import Agent, AgentRegistry
from string import Template
from copy import deepcopy
import os
import json
from threading import Thread
from abc import ABC, abstractmethod
import random


class ExecuteResult(NamedTuple):
    is_passing: bool
    feedback: str
    state: Tuple[bool]


class Executor(ABC):
    @abstractmethod
    def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
        ...

    @abstractmethod
    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        ...


class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


def function_with_timeout(func, args, timeout):
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError()
    else:
        return result_container[0]


# Use exec instead of eval, and capture the return value from locals()
def exec_and_return(code, globals_):
    locals_ = {}
    try:
        exec(code, globals_, locals_)
        var = code.split('\n')[-1].strip()
        if 'print(' in var:
            var = var.split('print(')[-1].split(')')[0]
        if var in locals_:
            return locals_[var]
        else:
            return None  # Return None if variable not found
    except Exception as e:
        return f"Error occurred: {e}"


class PyExecutor(Executor):
    def execute(self, codes: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{code}' for code in codes]

        is_passing = []
        execution_outputs = []
        for i in range(len(func_test_list)):
            try:
                # output = function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                output = function_with_timeout(exec_and_return, (func_test_list[i], globals()), timeout)
                execution_outputs.append(f"Execution status: success\nCode output: {output}")
                is_passing.append(True)
            except Exception as e:
                output = str(e)
                execution_outputs.append(f"Execution status: failed\nCode output: {output}")
                is_passing.append(False)

        return is_passing, execution_outputs

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """

        code = f"""{func}

{test}

check({name})
    """
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False


@AgentRegistry.register('MathSolver')
class MathSolver(Agent):
    def __init__(self, id: str | None = None, role: str = "", pretrain: str = "", agent_workflow: str = "", prompt: list = None, is_reasoning_model: bool = False, shuffle_responses: bool = False):
        super().__init__(id, role, pretrain, agent_workflow, is_reasoning_model)
        self.pretrain = pretrain
        self.role = role
        self.agent_workflow = agent_workflow
        self.prompt = prompt
        self.is_reasoning_model = is_reasoning_model
        self.shuffle_responses = shuffle_responses
        print(self.prompt)

    def _process_inputs(self, raw_inputs, spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], turn_id: int, **kwargs) -> List[Any]:
        num_prompt = len(raw_inputs)
        if 'system_prompt' in self.prompt[self.role]:
            system_prompt = [self.prompt[self.role]['system_prompt']] * num_prompt
        else:
            system_prompt = [""] * num_prompt
        user_prompt = raw_inputs
        turn_id = str(turn_id)

        for i in range(num_prompt):
            previous_responses = {}
            for node_id, info in spatial_info.items():
                if len(info["output"]):
                    previous_responses[node_id] = {"role": info["role"], "output": info["output"][i]}
            for node_id, info in temporal_info.items():
                if len(info["output"]):
                    previous_responses[node_id] = {"role": info["role"], "output": info["output"][i]}

            prompt_data = {'question': user_prompt[i]}
            if self.agent_workflow in ['multi-agents-debate', 'mixture-of-agents']:
                responses_str = ""
                previous_responses = [v["output"] for k, v in previous_responses.items()]
                if self.shuffle_responses:
                    random.shuffle(previous_responses)
                for id, agent_response in enumerate(previous_responses):
                    responses_str += f"Agent {id + 1} response: {agent_response}\n"
                prompt_data['responses_str'] = responses_str
                if 'responses_str' in self.prompt[self.role][turn_id] and not responses_str:
                    print(f"Warning: No agent responses found for round {turn_id}: {responses_str}, Spatial info: {spatial_info}, Temporal info: {temporal_info}")
                user_prompt[i] = Template(self.prompt[self.role][turn_id]).substitute(deepcopy(prompt_data))
            else:  # chain-of-agents
                for key, agent_response in previous_responses.items():
                    prompt_data[agent_response['role']] = agent_response['output']
                user_prompt[i] = Template(self.prompt[self.role][turn_id]).substitute(deepcopy(prompt_data))

        return system_prompt, user_prompt

    def _execute(self, inputs: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], turn_id: int, **kwargs):
        system_prompt, user_prompt = self._process_inputs(inputs, spatial_info, temporal_info, turn_id)
        messages = []
        for i in range(len(user_prompt)):
            if system_prompt[i]:
                messages.append([{'role': 'system', 'content': system_prompt[i]}, {'role': 'user', 'content': user_prompt[i]}])
            else:
                messages.append([{'role': 'user', 'content': user_prompt[i]}])
        return messages

    def execute_code(self, code_outputs, **kwargs):
        is_passing, execution_outputs = PyExecutor().execute(code_outputs, timeout=10)

        return execution_outputs
