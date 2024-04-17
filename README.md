# CTG

Source code of “Reinforcement Learning with Token-level Feedback for Controllable Text Generation (NAACL 2024)”

If you encounter problems, feel free to contact me (wendili@hust.edu.cn).

Currently, codes of sentiment transformation experiments have been published. 
Next, I will try to sort out codes of double-aspect CTG as soon as possible.

## Single-attribute Control

- Train a attribute classifier.

  In sentiment transformation, we retrain a attribute classifier with SST-5. To run the recommendation part.

```python Sentiment/main_disc.py```

- Run Token-level RL

  To train a policy model, run

  ```python token_main.py --source_mode neutral --target_mode positive --reward_model {best checkpoint of your classifier} ```

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
  
