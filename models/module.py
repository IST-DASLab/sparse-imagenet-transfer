from abc import abstractmethod
import contextlib
import logging
from typing import Iterable, List, Optional, Dict
import tempfile

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Identity

try:
    from deepsparse import compile_model
    deepsparse_err = None
except Exception as err:
    compile_model = None
    deepsparse_err = err


class ONNXModuleWrapper(Module):
    def __init__(
        self,
        module: Module,
        input_shape: Iterable[int],
        exclude_names_pre: Iterable[str] = None,
        exclude_names_post: Iterable[str] = ("fc",),
        preserve_onnx: bool = False,
    ):
        super().__init__()
        self._orig_module = module
        self._input_shape = input_shape
        self._exclude_names_pre = exclude_names_pre
        self._exclude_names_post = exclude_names_post
        self._pre_module = None
        self._post_module = None
        self._preserve_onnx = preserve_onnx

        self._create_exclusions()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        self._remove_exclusions()
        state = self._orig_module.state_dict(destination, prefix, keep_vars)
        self._create_exclusions()

        return state

    def load_state_dict(
        self,
        state_dict: Dict[str, Tensor],
        strict: bool = True
    ):
        self._remove_exclusions()
        self._orig_module.load_state_dict(state_dict, strict)
        self._create_exclusions()
        self.create_engine()

    def forward(self, *args, **kwargs):
        if kwargs:
            raise ValueError(
                "ONNXModuleWrapper does not support kwargs, "
                "the dictionary ordering is not guaranteed for deepsparse list input,"
                f"given: {kwargs}"
            )

        for tens in args:
            if not isinstance(tens, Tensor):
                raise ValueError(
                    f"ONNXModuleWrapper does not support non Tensor inputs, "
                    f"given {tens}"
                )

        out = self.onnx_pre_forward(*args)
        out = self.onnx_forward(*out)
        out = self.onnx_post_forward(*out)

        if len(out) == 1:
            out = out[0]

        return out

    def onnx_pre_forward(self, *args) -> List[Tensor]:
        out = args

        if self._pre_module is not None:
            out = self._pre_module(*args)
            if isinstance(out, Tensor):
                out = [args]

        return out

    @abstractmethod
    def onnx_forward(self, *args) -> List[Tensor]:
        raise NotImplementedError()

    def onnx_post_forward(self, *args) -> List[Tensor]:
        out = args

        if self._post_module is not None:
            out = self._post_module(*args)
            if isinstance(out, Tensor):
                out = [out]

        return out

    @abstractmethod
    def create_engine(self):
        raise NotImplementedError()

    @contextlib.contextmanager
    def create_onnx(self) -> str:
        tmp_onnx = tempfile.NamedTemporaryFile(delete=not self._preserve_onnx)

        if self._preserve_onnx:
            logging.info(f"ONNXModuleWrapper: onnx file preserved at {tmp_onnx.name}")

        try:
            sample_batch = torch.randn(*self._input_shape)
            torch.onnx.export(
                self._orig_module.cpu(),
                sample_batch,
                tmp_onnx,
                strip_doc_string=True,
                verbose=False,
            )

            yield tmp_onnx.name
        finally:
            tmp_onnx.close()

    def _create_exclusions(self):
        self._pre_module = self._exclude(self._exclude_names_pre)
        self._post_module = self._exclude(self._exclude_names_post)

    def _remove_exclusions(self):
        self._revert_exclude(self._exclude_names_pre, self._pre_module)
        self._revert_exclude(self._exclude_names_post, self._post_module)
        self._pre_module = None
        self._post_module = None

    def _exclude(self, names: Iterable[str]) -> Optional[Sequential]:
        if not names:
            return None

        excluded = []

        for name in names:
            props = name.split(".")
            layer = self._orig_module

            for prop in props[:-1]:
                layer = getattr(layer, prop)

            excluded.append(getattr(layer, props[-1]))
            setattr(layer, props[-1], Identity())

        return Sequential(*excluded)

    def _revert_exclude(self, names: Iterable[str], module: Sequential):
        if not names:
            return

        for name, orig in zip(names, module):
            props = name.split(".")
            layer = self._orig_module

            for prop in props[:-1]:
                layer = getattr(layer, prop)

            setattr(layer, props[-1], orig)


class DeepSparseModuleWrapper(ONNXModuleWrapper):
    def __init__(self, module: Module, input_shape: Iterable[int]):
        if deepsparse_err:
            raise deepsparse_err

        super().__init__(module, input_shape)
        self._batch_size = list(input_shape)[0]
        self._engines = {}
        self.create_engine()

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        super().load_state_dict(state_dict, strict)
        self.create_engine()

    def create_engine(self):
        if self._engines:
            del self._engines

        self._engines = {
            self._batch_size: self._create_batch_engine(self._batch_size)
        }

    def onnx_forward(self, *args) -> List[Tensor]:
        batch_size = None

        for tens in args:
            batch_size = tens.shape[0]

            if batch_size not in self._engines:
                logging.warning(
                    f"DeepSparseModuleWrapper forward: batch_size given of "
                    f"{tens.shape[0]} does not match the constructed batch_size of "
                    f"{self._batch_size}, constructing new engine to support"
                )
                self._engines[batch_size] = self._create_batch_engine(batch_size)

        with torch.no_grad():
            tensors = [tens.detach().cpu() for tens in args]
            outputs = self._engines[batch_size]([tens.numpy() for tens in tensors])

            return [torch.from_numpy(out) for out in outputs]

    def _create_batch_engine(self, batch_size: int):
        with self.create_onnx() as onnx_file:
            logging.info(
                f"DeepSparseModuleWrapper: creating engine from onnx at {onnx_file} "
                f"for batch_size {batch_size}"
            )
            return compile_model(onnx_file, batch_size)
