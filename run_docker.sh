docker run -itd \
--gpus '"device=0"' \
-v `pwd`:/priorMDM \
--restart always \
-p 8019:8019 \
ralphhan/priormdm \
bash /priorMDM/run_server.sh
