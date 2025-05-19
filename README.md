# AgentSynth
AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents

See https://huggingface.co/datasets/sunblaze-ucb/AgentSynth for our data.

Below are instructions to run our pipeline:

### OSWorld

Please refer to https://github.com/xlang-ai/OSWorld for environment setup.

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


### Data
https://huggingface.co/datasets/sunblaze-ucb/AgentSynth
