FROM python:3.9.5

WORKDIR /usr/app/src/CBG_Fleming_Model

RUN pip3 install numpy==1.23.1 scipy==1.9.0 PyNN==0.10.0
RUN pip3 install NEURON==8.0
RUN pip3 install nrnutils==0.2.0

COPY ./Cortex_BasalGanglia_DBS_model/Updated_PyNN_Files/pynn-steady-state.patch ./
WORKDIR /usr/local/lib/python3.9
RUN patch -p1 < /usr/app/src/CBG_Fleming_Model/pynn-steady-state.patch

WORKDIR /usr/local/lib/python3.9/site-packages/pyNN/neuron/nmodl
RUN nrnivmodl

WORKDIR /usr/app/src/CBG_Fleming_Model

COPY ./Cortex_BasalGanglia_DBS_model/*.txt ./

COPY ./Cortex_BasalGanglia_DBS_model/*.c ./
COPY ./Cortex_BasalGanglia_DBS_model/*.mod ./
COPY ./Cortex_BasalGanglia_DBS_model/*.o ./
COPY ./Cortex_BasalGanglia_DBS_model/*.html ./
COPY ./Cortex_BasalGanglia_DBS_model/steady_state_docker.bin ./steady_state.bin
RUN nrnivmodl

COPY ./Cortex_BasalGanglia_DBS_model/*.py ./
COPY ./Cortex_BasalGanglia_DBS_model/*.npy ./

ENTRYPOINT ["python3", "/usr/app/src/CBG_Fleming_Model/run_CBG_Model_IFT.py"]