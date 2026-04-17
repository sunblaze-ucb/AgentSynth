# AgentSynth
[![ArXiv](https://img.shields.io/badge/arXiv-2506.14205-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.14205)
[![Hugging Face](https://img.shields.io/badge/Data-AgentSynth-orange?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/sunblaze-ucb/AgentSynth)
[![Website](https://img.shields.io/badge/Website-AgentSynth-blue?style=flat&logo=safari&logoColor=white)](https://sunblaze-ucb.github.io/agentsynth_web/)


## AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents [ICLR 2026]

![](Pipeline4.png?raw=true)

Below are instructions to run our pipeline:

### Data Generation in OSWorld

Please refer to https://github.com/xlang-ai/OSWorld for environment setup.

To install requirements:
```
pip install -r requirements.txt
```

To generate tasks and trajectories:
```
python generate_and_save_traces_persona.py
```

### InSTA

First clone the repository into the insta folder using the following command:
```
cd insta_data
git clone https://github.com/data-for-agents/insta.git
cd insta
git checkout $(git rev-list --max-count=1 --before="2025-05-12" main)
cd ..
```
Then, if you have not run InSTA before, pull their Docker image:
```
docker pull brandontrabucco/insta-browser-environment
```
Load your secret API keys in a file called `insta_data\secrets.json`. Finally, run:
```
python combined_task_generation_new.py --env insta
```
For troubleshooting your InSTA installation, refer to their repository at https://github.com/data-for-agents/insta/tree/main.


### 📄 Citation

If you use AgentSynth in your research, please cite our paper:
```bibtex
@article{xie2025agentsynth,
  title={AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents},
  author={Xie, Jingxu and Xu, Dylan and Zhao, Xuandong and Song, Dawn},
  journal={arXiv preprint arXiv:2506.14205},
  year={2025}
}
```
