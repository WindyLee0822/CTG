# TOLE: Reinforcement Learning with Token-level Feedback for Controllable Text Generation

Source code of “Reinforcement Learning with Token-level Feedback for Controllable Text Generation (NAACL 2024)”

If you encounter problems, feel free to contact me (wendili@hust.edu.cn).


## Multi-attribute Control

- Train a weigher.

  When we have one sentiment scorer and one topic scorer, we need to train a weigher to weight them.

```python weigher.py --sent_scorer_path <path of your best sentiment classifier> --topic_scorer_path <path of your best topic classifier>```

- Run Token-level RL

  To train a policy model, run

  ```python token_main.py --sent_reward_model {best checkpoint of your sentiment classifier} --topic_reward_model {best checkpoint of your topic classifier} --weigher_ckpt {final checkpoint of your weigher}```

## Citation
  If you find our research helpful, please kindly cite our paper!

```bibtex
@article{li2024reinforcement,
  title={Reinforcement Learning with Token-level Feedback for Controllable Text Generation},
  author={Li, Wendi and Wei, Wei and Xu, Kaihe and Xie, Wenfeng and Chen, Dangyang and Cheng, Yu},
  journal={arXiv preprint arXiv:2403.11558},
  year={2024}
}
```
  
