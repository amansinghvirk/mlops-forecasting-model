python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python mlmodel/src/train.py --execution_name "Validation Sets" --experiments_list "params/Validation Sets.yaml"
python mlmodel/src/train.py --execution_name "Variables Set" --experiments_list "params/Variables Set.yaml"
cd mlmodel/src
uvicorn experiment_api:app --port $2 --reload