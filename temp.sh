chmod 600 /data/coding/id_rsa
git -c core.sshCommand="ssh -i /data/coding/id_rsa -F /dev/null" clone git@github.com:rrrsj/cuda-kernel.git
nvcc -arch=sm_50 hello_world.cu -o ../hello_world