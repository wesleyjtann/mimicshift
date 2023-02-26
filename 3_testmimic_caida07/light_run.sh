mkdir $1
mkdir $1/artefact
mkdir $1/result

# python preprocess.py light_config.yaml
python light_train.py light_config.yaml
python light_inference_seq.py light_config.yaml
python light_evaluation_seq.py light_config.yaml