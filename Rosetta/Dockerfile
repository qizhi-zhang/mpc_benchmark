FROM morse-registry-vpc.cn-shanghai.cr.aliyuncs.com/rd/rosetta:v20220314

# compile the latest version
RUN cd /home/ && git clone --recurse https://github.com/LatticeX-Foundation/Rosetta.git
RUN cd /home/Rosetta/ && rm -y build && ./rosetta.sh compile --enable-protocol-mpc-securenn && ./rosetta.sh install
# RUN cd /home/Rosetta/ && ./rosetta.sh compile --enable-protocol-mpc-securenn && ./rosetta.sh install
# RUN cd /home/Rosetta/example/tutorials/code && ./tutorials.sh rtt linear_regression_reveal && tail -n 60 log/linear_regression_reveal-0.log

ADD ./app  /app