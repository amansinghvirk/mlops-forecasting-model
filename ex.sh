python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python mlmodel/src/train.py --execution_name "Validation Sets" --experiments_list "/c/mydata/work/mlproject/final_project/models/model_type_a_stores/params/Validation Sets.yaml"
python mlmodel/src/train.py --execution_name "Variables Set" --experiments_list "/c/mydata/work/mlproject/final_project/models/model_type_a_stores/params/Variables Set.yaml"
cd mlmodel/src
uvicorn experiment_api:app --port $2 --reload
