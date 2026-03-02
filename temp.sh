chmod 600 /data/coding/id_rsa
git -c core.sshCommand="ssh -i /data/coding/id_rsa -F /dev/null" clone git@github.com:rrrsj/cuda-kernel.git
git -c core.sshCommand="ssh -i /data/coding/id_rsa -F /dev/null" push
git config --global user.email "shj_ren@foxmail.com"
git config --global user.name "Shijie Ren"