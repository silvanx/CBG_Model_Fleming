FROM python:3.9.5

ENV PYTHONUNBUFFERED=1

WORKDIR /usr/app/src/CBG_Fleming_Model

RUN pip3 install numpy==1.23.1 scipy==1.9.0 PyNN==0.10.0
RUN pip3 install NEURON==8.0
RUN pip3 install nrnutils==0.2.0
RUN pip3 install pyyaml

COPY ./Cortex_BasalGanglia_DBS_model/Updated_PyNN_Files/pynn-steady-state.patch ./
WORKDIR /usr/local/lib/python3.9
RUN patch -p1 < /usr/app/src/CBG_Fleming_Model/pynn-steady-state.patch

WORKDIR /usr/local/lib/python3.9/site-packages/pyNN/neuron/nmodl
RUN nrnivmodl


RUN echo "deb http://archive.debian.org/debian buster main contrib non-free" > /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y install openmpi-bin=3.1.3-11
RUN pip3 install mpi4py==3.1.4
RUN apt-get -y install time
RUN pip3 install debugpy cerberus

WORKDIR /usr/app/src/CBG_Fleming_Model

COPY ./Cortex_BasalGanglia_DBS_model/burst_data/*.txt ./burst_data/
COPY ./Cortex_BasalGanglia_DBS_model/network_structure/*.txt ./network_structure/
COPY ./Cortex_BasalGanglia_DBS_model/network_structure/*.npy ./network_structure/
COPY ./Cortex_BasalGanglia_DBS_model/neuron_mechanisms/*.mod ./neuron_mechanisms/

WORKDIR /usr/app/src/CBG_Fleming_Model/neuron_mechanisms

RUN nrnivmodl


WORKDIR /usr/app/src/CBG_Fleming_Model

COPY ./Cortex_BasalGanglia_DBS_model/*.py ./
COPY ./Cortex_BasalGanglia_DBS_model/*.npy ./
COPY ./Cortex_BasalGanglia_DBS_model/configs/*.yml ./configs/

ENTRYPOINT ["mpirun", "--allow-run-as-root", "-np", "6", "python3", "/usr/app/src/CBG_Fleming_Model/run_model.py"]
CMD ["/usr/app/src/CBG_Fleming_Model/configs/conf_zero_4s.yml"]
