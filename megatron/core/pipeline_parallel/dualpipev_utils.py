import queue
from typing import List, Callable, Optional

import torch
from torch.autograd import Variable


def overlapped_forward_backward(
    module0: torch.nn.Module,
    inputs0: List[torch.Tensor],
    labels0: Optional[List[torch.Tensor]],
    loss_masks0: Optional[List[torch.Tensor]],
    loss1: Optional[torch.Tensor],
    outputs1: Optional[List[torch.Tensor]],
    output_grads1: Optional[List[torch.Tensor]],
    forward_step_func: Callable,
    is_last_stage0: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    You should implement custom forward-backward overlap strategy.
    The code below is just an example.
    """
    if len(inputs0) == 1:
        from megatron.core.utils import get_attr_wrapped_model
        set_input_tensor = get_attr_wrapped_model(module0, "set_input_tensor")
        set_input_tensor(inputs0)
    if is_last_stage0:
        inputs0_with_labels_loss_masks = list(inputs0)
        inputs0_with_labels_loss_masks.append(labels0)
        inputs0_with_labels_loss_masks.append(loss_masks0)
        outputs0, loss_func = forward_step_func(inputs0_with_labels_loss_masks, module0)
    else:
        outputs0, loss_func = forward_step_func(inputs0, module0)
    outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
    if is_last_stage0:
        loss0 = loss_func(outputs0[0])[0]
    else:
        loss0 = None

    if loss1 is not None:
        loss1.backward()
        loss1.detach_()
    else:
        run_backward(outputs1, output_grads1)

    return outputs0, loss0


class WeightGradStore:

    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()


def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    kwargs = dict(
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True,
    )
    Variable._execution_engine.run_backward(tensors, grad_tensors, **kwargs)


def chunk_tensor(x, chunks, dim):
    if x is None:
        return [None for _ in range(chunks)]
    return x.tensor_split(chunks, dim=dim)


def cat_tensor(x, dim):
    if (isinstance(x, tuple) or isinstance(x, list)):
        if len(x) == 1:
            return x[0]
        elif x[0] is None:
            assert all(y is None for y in x)
            return None
    return torch.cat(x, dim=dim)


def scatter(inputs, chunks, dim):
    assert isinstance(inputs, (torch.Tensor, tuple, list))
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    assert all(x is None or isinstance(x, torch.Tensor) for x in inputs)
    inputs = [chunk_tensor(x, chunks, dim) for x in inputs]
    microbatches = [microbatch for microbatch in zip(*inputs)]
    if len(microbatches) == 0:
        microbatches = [() for _ in range(chunks)]
    return microbatches


def gather(micro_outputs, dim):
    assert isinstance(micro_outputs[0], (torch.Tensor, tuple, list))
    if isinstance(micro_outputs[0], torch.Tensor):
        micro_outputs = [(x,) for x in micro_outputs]
    outputs = [x for x in zip(*micro_outputs)]
    outputs = tuple(cat_tensor(x, dim=dim) for x in outputs)
    return outputs
