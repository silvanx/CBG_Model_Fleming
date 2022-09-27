FROM python:3.9.5

WORKDIR /usr/app/src/CBG_Fleming_Model

COPY ./Cortex_BasalGanglia_DBS_model/*.txt ./
COPY ./Cortex_BasalGanglia_DBS_model/*.py ./
COPY ./Cortex_BasalGanglia_DBS_model/*.c ./
COPY ./Cortex_BasalGanglia_DBS_model/*.mod ./
COPY ./Cortex_BasalGanglia_DBS_model/*.o ./
COPY ./Cortex_BasalGanglia_DBS_model/*.html ./
COPY ./Cortex_BasalGanglia_DBS_model/*.bin ./
COPY ./Cortex_BasalGanglia_DBS_model/*.bin ./
COPY ./Cortex_BasalGanglia_DBS_model/*.npy ./
COPY ./Cortex_BasalGanglia_DBS_model/Updated_PyNN_Files/ ./


RUN pip3 install numpy==1.23.1 scipy==1.9.0 PyNN==0.10.0
RUN pip3 install NEURON==8.0
RUN pip3 install nrnutils==0.2.0

RUN nrnivmodl

