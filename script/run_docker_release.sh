docker run -itd \
--name motion-search \
-v `pwd`:/priorMDM \
--restart always \
-p 6399:6399 \
ralphhan/commander \
bash /priorMDM/script/run_server_release.sh
