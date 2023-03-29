import logging
from cv2 import log
import torch
from bisect import bisect_right
from detectron2.engine import HookBase

logger = logging.getLogger(__name__)

class BoxTeacherEMAHook(HookBase):

    def __init__(self, interval=1, momentum=0.999, warmup=0, decay_factor=0.1, decay_intervals=None, update_fn="v1"):
        super().__init__()
        self.interval = interval
        self.momentum = momentum
        self.warmup = warmup
        self.decay_factor = decay_factor
        self.decay_intervals = decay_intervals
        self.momentum_update_fn = self.momentum_update

    def before_train(self):
        cur_iter = self.trainer.iter
        if cur_iter == 0:
            self.init_param(self.trainer.model.module)
            logger.info("init teacher with student")
        else:
            logger.info("init teacher with loaded checkpoints")

    def init_param(self, model):
        with torch.no_grad():
            for src_parm, tgt_parm in zip(
                model.student.state_dict().values(), model.teacher.state_dict().values()
            ):
                tgt_parm.data.copy_(src_parm.data)

    def param_diff(self, model):
        param_norm = []
        with torch.no_grad():
            for (src_name, src_parm), tgt_parm in zip(
                model.student.state_dict().items(), model.teacher.state_dict().values()
            ):
                if "_iter" in src_name or "sizes_of_interest" in src_name or "num_batches_tracked" in src_name:
                    continue
                param_norm.append(torch.norm((src_parm - tgt_parm).float()))
            param_norm = torch.stack(param_norm).mean().item()
            self.trainer.storage.put_scalar("param_diff", param_norm)

    def momentum_update(self, model, momentum):
        with torch.no_grad():
            for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
                model.student.named_parameters(), model.teacher.named_parameters()
            ):
                tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

    def before_step(self):
        cur_iter = self.trainer.iter
        if cur_iter % self.interval != 0:
            return
        model = self.trainer.model.module
        momentum = min(
            self.momentum, 1 - (1 + self.warmup) / (cur_iter + 1 + self.warmup)
        )
        self.trainer.storage.put_scalar("ema_momentum", momentum)

        self.momentum_update_fn(model, momentum)
        self.param_diff(model)

    def after_step(self):
        cur_iter = self.trainer.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, cur_iter
        )
