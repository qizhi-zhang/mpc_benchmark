# base image
# FROM registry.cn-hangzhou.aliyuncs.com/dtunion/morsebase:vsmodel1
#FROM registry.cn-hangzhou.aliyuncs.com/dtunion/morsetfe:v2.4
# FROM registry.cn-hangzhou.aliyuncs.com/dtunion/morsetfe:v3.16
# FROM registry.cn-hangzhou.aliyuncs.com/dtunion/morsebase:vpython3base
#FROM reg.docker.alibaba-inc.com/morse/morsebase:v20210408_38
#FROM continuumio/miniconda3
FROM realite/python3.6



# add appliction code
ADD ./app /app


# requirements
#RUN conda create -n tfe python=3.6
#RUN conda activate tfe
# RUN conda install tensorflow notebook

RUN python3 -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install tf-encrypted -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install pandas -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install scikit-learn -i https://mirrors.aliyun.com/pypi/simple/
# RUN apt-get install curl -y



# nginx
#RUN mkdir -p /app/conf
# ADD ./conf/dataleaguelocal_nginx.conf /app/conf
# ADD ./conf/docker-entry-tfe.sh  /app/
# ADD ./conf/tfe_uwsgi.ini  /app/conf

ENTRYPOINT ["/Player/docker-entry-tfe.sh"]

