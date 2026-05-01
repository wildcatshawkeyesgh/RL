**Project: Deep Reinforcement Learning for IADS Penetration Planning Using a Transformer World Model**

**How to setup and run**

clone this directory to wherever you want to run it.

Then run uv sync.

To run just a single run use this command and change the name to whatever you want it to be:

uv run python -m scripts.ppo_impl --run-name init_log_mean

To create a video use this command uv run python -m scripts.compare_checkpoints \
    checkpoints/init_log_mean/checkpoint_000655360.onnx \ 
checkpoints/init_log_mean/checkpoint_003276800.onnx \
checkpoints/init_log_mean/checkpoint_007864320.onnx \
checkpoints/init_log_mean/checkpoint_009994240.onnx \
--seed 42 \
    --output runs/<run-name>_progression.mp4

To run the tensorboard use:

uv run tensorboard --logdir runs 

To run a simple study use:./scripts/run_studies.sh     "gamma-0.995:--gamma 0.995"     "logstd--0.8:--init-log-std -0.8"     "hidden-3:--hidden-layers 3"

**Editing Variables and Environment**

All of the rewards parameters and the enrionment variables can be set in the iads folder on the default.yaml file.

The hyperparameters for the neural network/reinforcement learning can be found in ppo/config.yaml

It would be intersting to run a real study on this with MLFlow and Ax instead of just viewing the data with tensorboard.


**Dataset Description**

This project uses reinforcement learning rather than a dataset. The training data is generated through agent environment interaction within a custom environment. Each episode of the simulation produces state,action, and reward values that serve as the training data for the neural network.



The simulation models a missile strike planning scenario against an Integrated Air Defense System (IADS). The attacking blue model controls real strike missiles and decoy missiles attempting to penetrate a network of radar sites and SAM batteries to destroy targets. The defender is controlled by a simple script. The red model has randomness to it in terms of the positions of the SAMS, early warning radars, and targets. They are biased so that the targets are always behind the SAMS and EWR, while having the EWR in the front of the SAMS.

**Observation Space**

Each timestep produces a state observation vector containing approximately 150-200 features:

- Missile states (per missile): position (x, y, z), velocity (vx, vy), speed, fuel remaining, altitude, tracking status, number of tracking radars, evasion status, missile type (real/decoy) — ~12 features × 8 missiles = 96 features
- SAM battery states: position (x, y), engagement range, interceptors remaining, cooldown status — ~5 features × 8 SAMs = 40 features
- Radar states: position (x, y), detection range, alive status — ~4 features × 4 radars = 16 features
- Target states: position (x, y), alive status — ~3 features × 3 targets = 9 features
- Interceptor states (active): position (x, y, z), target missile ID — variable, up to ~4 features × 10 active interceptors
- Global state: timestep, number of missiles in flight, number of targets remaining

**Reward Signal**

The agent receives a scalar reward signal based on mission outcomes:
- Exploration of new state +# (with decay over time)
- Target destroyed: +10.0
- Real missile intercepted: -2.0
- All targets destroyed bonus: +5.0 (or a multiplyer for each target destroyed so one target destroyed is +10, two is +25, and three is +45)


There is potential for reward shaping by using the following:
- Decoy absorbs an interceptor: +1.5 (trains decoy to be sacrifical lamb)
- Targets destroyed at the same time: +1.0 (trains simultenous time on target)
- Successful Evasion: +1.0 (probably requires runs out of fuel to be equal to or greater negative value)
- Runs out of fuel: -1.0
- Only one missile launched at multiple missiles flying in tight formation:+0.5 (tricky to implement but I added a small radar resolution where when missiles and decoys are close enough toghether they will appear as one single missile with the hope that flying in formation improves penetration)
- 

**Action Space**

Each timestep the agent outputs a continuous action vector containing heading change and altitude change per missile, plus evasion toggles.


**Problem Statement**

The problem is given knowledge of a defended airspace (radar and SAM positions, ranges, capabilities), how should an attacker optimally route, time, and coordinate a mixed salvo of real and decoy missiles to maximize the probability of destroying defended targets? This involves learning tactical behaviors such as formation flying to mask radar signatures, decoy employment to waste defender interceptors, time-on-target coordination, and evasive maneuvering to survive interceptor engagements.
While this simulation is realtively light weight and should easily run fast enough for many epochs not all simulation software can be run as quickly. AFISM (Advanced Framework for Simulation, Integration and Modeling) is the standard tool for this type of analysis used by the DoD. While AFSIM models can contain decently high fidelity the simulations can be incredibly slow. In order to effectively use reinforcment learning the total number of simulation runs needs to be limited. That's why for this project I want to investigate using a
transformer model like that used in TransDreamer and TransDreamerV3. These sample efficent models combine a trasnformer to learn the model with a soft-actor critic for offline play.
Due to the dificulty of completing this before the project deadline my backup plan is to try and use ppo to evaluate differnet reward functions and the impact thehy have on convergence time and overall ability to win the game. I had some other ideas to extend the transdreamer architecure such as
implementing a an event based replan by the agent. Some new research is using the transdreamer to update after each time step and using the world model to repredict the best course of action based on what haas gone on
in the past timesteps I think it would intersting to do that based on specific scenario events rather than after every time step. An example could be if a missile is detected to be tracking maybe the overall best course of action is to 
allow that missile to be destroyed but do so by veering away from other missiles. This also leads into another idea I had which is what if the agent discovers the original intelligence on the red laydown 
was wrong and there is a differnet type of SAM system there or there is a SAM system somewhere that wasn't initially thought to be there. Those events could trigger a replan.


**Current State and Course of Action**

Currently the game runs and can be played and the simulation engine should be able to be ran headless on a the Linux HPC. The game itself still needs to be tuned to a point where I am able to beat it. If the agent turns out to be much better than me then I will readjust the difficulty back up.
I am planning on using a gymnasium environemnt which is currently being built. I am planning on using TorchRL which is a sublibrary of PyTorch. I have been doing research and gathering my thoughts on whether I want to attmept to create a mini transdreamer architecture and whether it would be feasible to do in 
our time frame. If I did go that route I would likely not have a lot of time to do a full evaluation of reward functions and hyperparameters. I do believe a multi layer perceptron would mostlikely yield
the same or better results but I am merely intersted in using a transfromer like this in combination of reinforcement learning.

