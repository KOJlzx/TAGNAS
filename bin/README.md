# Training and evaluation commands

Run commands from the repository root after installing the environment with
`uv sync --locked`. The root shell runners configure training and
forward arguments to these Python entry points.

| Training task | Shell runner | Python entry point |
| --- | --- | --- |
| TAGNAS search and retraining | [`run_sys.sh`](../run_sys.sh) | [`async_coexec_train.py`](async_coexec_train.py) |
| ConGraT baseline | [`run_clip.sh`](../run_clip.sh) | [`congrat_train.py`](congrat_train.py) |
| DDP supernet baseline | [`run_ddp.sh`](../run_ddp.sh) | [`ddp_train.py`](ddp_train.py) |
| ZeRO-Offload supernet baseline | [`run_deepspeed.sh`](../run_deepspeed.sh) | [`zero_offload_train.py`](zero_offload_train.py) |
| UniTG-style baseline | [`run_unitg.sh`](../run_unitg.sh) | [`unitg_train.py`](unitg_train.py) |

Inspect the arguments without starting training:

```bash
uv run --no-sync python bin/async_coexec_train.py --help
uv run --no-sync python bin/congrat_train.py --help
uv run --no-sync python bin/ddp_train.py --help
uv run --no-sync python bin/zero_offload_train.py --help
```

| Evaluation task | Entry point |
| --- | --- |
| TAGNAS link prediction | [`eval_link_pred_distributed.py`](eval_link_pred_distributed.py) |
| ConGraT link prediction | [`eval_link_pred_clip_ddp.py`](eval_link_pred_clip_ddp.py) |
| DDP / ZeRO-Offload link prediction | [`eval_link_pred_ddp.py`](eval_link_pred_ddp.py) |
| UniTG-style link prediction | [`eval_link_pred_unitg.py`](eval_link_pred_unitg.py) |
| Inspect architecture weights | [`inspect_alphas.py`](inspect_alphas.py) |

The older Lightning workflow uses `trainer.py`, `eval.py`, and `score.py`.
`congrat_single_train.py` retains the original single-device training utility;
its main block runs a predefined dataset/model sweep. The configurable ConGraT
interface is `run_clip.sh`.
