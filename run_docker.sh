docker run -itd \
--gpus '"device=0"' \
-v `pwd`:/PriorMDM \
--restart always \
-p 8019:8019 \
ralphhan/priormdm \
bash /PriorMDM/run_server.sh
