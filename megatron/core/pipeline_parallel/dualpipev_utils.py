import queue
import threading
from typing import List, Callable, Optional

import torch
from torch.autograd import Variable
from megatron_patch.template.helper import attention_forward

comm_stream = torch.cuda.Stream(device="cuda")

def overlapped_forward_backward(
    module0: torch.nn.Module,
    inputs0: List[torch.Tensor],
    labels0: Optional[List[torch.Tensor]],
    loss_masks0: Optional[List[torch.Tensor]],
    module1: torch.nn.Module,
    loss1: Optional[torch.Tensor],
    outputs1: Optional[List[torch.Tensor]],
    output_grads1: Optional[List[torch.Tensor]],
    is_last_stage0: bool,
    chunk_id0: int,
    chunk_id1: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    You should implement custom forward-backward overlap strategy.
    The code below is just an example.
    """
    # Prepare inputs for forward
    if len(inputs0) == 1:
        from megatron.core.utils import get_attr_wrapped_model
        set_input_tensor = get_attr_wrapped_model(module0, "set_input_tensor")
        set_input_tensor(inputs0)
    if is_last_stage0:
        inputs0_with_labels_loss_masks = list(inputs0)
        inputs0_with_labels_loss_masks.append(labels0)
        inputs0_with_labels_loss_masks.append(loss_masks0)
        final_inputs0 = (inputs0_with_labels_loss_masks, module0, chunk_id0)
    else:
        final_inputs0 = (inputs0, module0, chunk_id0)

    # Prepare inputs for backward
    if loss1 is not None:
        final_inputs1 = (loss1, None, chunk_id1)
    else:
        final_inputs1 = (None, output_grads1, chunk_id1)

    global comm_stream

    # stage 0
    loss_func = attention_forward(*final_inputs0)
    
    # stage 1
    def stage1():
        with torch.cuda.stream(comm_stream):
            module0.dispatch()
    thread = threading.Thread(target=stage1)
    thread.start()
    module1.moe_post_backward(*final_inputs1)
    thread.join()
    torch.cuda.current_stream().wait_stream(comm_stream)

    # stage 2
    def stage2():
        with torch.cuda.stream(comm_stream):
            module1.combine_backward()
    thread = threading.Thread(target=stage2)
    thread.start()
    module0.moe_forward()
    thread.join()
    torch.cuda.current_stream().wait_stream(comm_stream)

    # stage 3
    def stage3():
        with torch.cuda.stream(comm_stream):
            module0.combine()
    thread = threading.Thread(target=stage3)
    thread.start()
    module1.moe_backward()
    thread.join()
    torch.cuda.current_stream().wait_stream(comm_stream)

    # stage 4
    def stage4():
        with torch.cuda.stream(comm_stream):
            module1.dispatch_backward()
    thread = threading.Thread(target=stage4)
    thread.start()
    outputs0 = module0.moe_post()
    thread.join()
    torch.cuda.current_stream().wait_stream(comm_stream)

    # stage 5
    module1.attention_backward()

    # post-process
    outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
    if is_last_stage0:
        loss0 = loss_func(outputs0[0])[0]
    else:
        loss0 = None

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
