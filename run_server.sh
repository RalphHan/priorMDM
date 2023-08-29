cd /priorMDM
rm body_models;ln -s /body_models .
rm save;ln -s /save .
rm glove;ln -s /glove .
/opt/conda/envs/PriorMDM/bin/uvicorn main:app --host 0.0.0.0 --port 8019 --workers 2
