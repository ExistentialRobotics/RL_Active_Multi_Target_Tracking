# Landmark-based mapping
This repository is a PyTorch implementation for paper ***[Policy Learning for Active Target Tracking over Continuous SE(3)
Trajectories](https://arxiv.org/pdf/2212.01498.pdf)***
in L4DC 2023. Authors: [Pengzhi Yang](https://pengzhi1998.github.io/), [Shumon Koga](https://shumon0423.github.io/), [Arash Asgharivaskasi](https://arashasgharivaskasi-bc.github.io/), 
[Nikolay Atanasov](https://natanaso.github.io/).
If you are using the code for research work, please cite:
```
@inproceedings{yang2023l4dc,
  title={Policy Learning for Active Target Tracking over Continuous SE(3) Trajectories},
  author={Yang, Pengzhi and Koga, Shumon and Asgharivaskasi, Arash and Atanasov, Nikolay},
  booktitle={Learning for Dynamics and Control (L4DC)},
  year={2023}
}
```

[//]: # (Design a RL policy which drives the agent to localize and update the landmarks' positions with fixed steps in a randomized )

[//]: # (environment.)

[//]: # (The yaml files are borrowed from this great repo: https://github.com/ehfd/docker-nvidia-glx-desktop.git and)

[//]: # (https://ucsd-prp.gitlab.io/userdocs/running/gui-desktop/)

## Intro
This paper proposes a novel model-based policy gradient algorithm for tracking dynamic targets
using a mobile robot, equipped with an onboard sensor with limited field of view. The task is to
obtain a continuous control policy for the mobile robot to collect sensor measurements that reduce
uncertainty in the target states, measured by the target distribution entropy. We design a neural
network control policy with the robot SE(3) pose and the mean vector and information matrix
of the joint target distribution as inputs and attention layers to handle variable numbers of targets.
We also derive the gradient of the target entropy with respect to the network parameters explicitly,
allowing efficient model-based policy gradient optimization.

## Installation
Clone the repository and ```cd``` into it,
```
conda create -n landmark_mapping python==3.8 -y
conda activate landmark_mapping
pip install -r requirements.txt
```

## Run and test
```cd``` into the ```model_based_active_mapping``` directory and:
```
python run_model_based_training.py
python run_model_based_test.py
```

## Experimental Results
<div style="display:flex;">
  <img src="https://github.com/ExistentialRobotics/RL_Active_Multi_Target_Tracking/blob/main/3landmarks.gif" width="30%">
  <img src="https://github.com/ExistentialRobotics/RL_Active_Multi_Target_Tracking/blob/main/5landmarks.gif" width="30%">
  <img src="https://github.com/ExistentialRobotics/RL_Active_Multi_Target_Tracking/blob/main/7landmarks.gif" width="30%">
</div>
