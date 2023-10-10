if [ $# -eq 0 ]; then
    docker run -itd \
      --gpus '"device=0"' \
      -v `pwd`:/priorMDM \
      --restart always \
      -p 8019:8019 \
      ralphhan/priormdm \
      bash /priorMDM/$0 1
    exit 0
fi

cd /priorMDM
rm body_models;ln -s /body_models .
rm save;ln -s /save .
rm glove;ln -s /glove .
/opt/conda/envs/PriorMDM/bin/uvicorn main:app --host 0.0.0.0 --port 8019 --workers 2
