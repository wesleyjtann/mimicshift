# Poisoning Online Learning Filters by Shifting on the Move


We introduce a poisoning attack that takes a contextual generative approach to generate shifting malicious traffic, studying its effects on online deep-learning DDoS filters. We investigate an adverse scenario where the attacker is “crafty”, switching profiles during attacks and generating erratic attack traffic. 

![shiftingatk](/image/atkddos-edit2.png)


## Cite

Please cite our paper if you find this code useful for your own work:

```
@article{tann2021poisoning,
	author={Wesley Joon-Wie Tann and Ee-Chien Chang},
	title= {Poisoning Online Learning Filters by Shifting on the Move},
	year={2021},
	journal={arXiv preprint, arXiv:2107.12612},
}
```


## Prerequisites

Create a Conda environment from the environment.yml file.

For generating the attack traffic.
```
conda env create -f mimicshift_env.yml
```
For evaluating the online DDoS defense. 
```
conda env create -f ddos_env.yml
```


## Data

Our datasets are released on Google Drive.

`https://drive.google.com/drive/folders/1-9d3z_mv-Cn2ZsWrRg106wVe73rHo2a3?usp=sharing`


## Running the experiments

All code are in the folders:
* /3_mimicshift
* /3_testmimic_caida07
* /3_testmimic_cicFriday
* /3_testmimic_cicFriday


* To run the experiments, execute the scripts in the respective folders.

_Step 1_
\[3_testmimic_xxx\]	Preprocess data by running ```sh light_run_xxx.sh; preprocess.py```

_Step 2_		
2a. \[3_mimicshift (GAN)	Train generator. ```python mimic(cond_train)1.py```
2b. \[3_mimicshift (GAN)	Generate mimic atk data. ```mimic_load-1.ipynb```
		
_Step 3_
3a.	\[3_testmimic_xxx\]	Eval N/D filters. ```sh light_run_xxx.sh```
3b.	\[3_testmimic_xxx\] Eval iter classifier. ```online_classifier.ipynb, online_classifier_inf.ipynb```
3c.	\[3_testmimic_xxx\] (atk with/without N. no counter)	Eval iter classifier. ```online_classifier.ipynb, online_classifier_inf.ipynb```
3d.	\[3_testmimic_xxx\] (countermeasure)	saving all countermeasure results. ```eval_it.ipynb```


