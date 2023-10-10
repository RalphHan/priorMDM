if [ $# -eq 0 ]; then
    docker run -itd \
      --name motion-search \
      -v `pwd`:/priorMDM \
      --restart always \
      -p 6399:6399 \
      ralphhan/commander \
      bash /priorMDM/$0 1
    exit 0
fi

cd /priorMDM
/opt/conda/bin/uvicorn main_release:app --host 0.0.0.0 --port 6399 --workers 8
