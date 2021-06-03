FROM  continuumio/miniconda3
ADD ./  /finalProject
WORKDIR finalProject
RUN [ "conda","update", "-n","base", "-c", "defaults", "conda" ]
RUN [ "conda", "install" ,"mamba","-c", "conda-forge" ]
RUN [ "mamba", "install", "-c", "conda-forge","pandas", "dash","nltk", "numpy","pillow","matplotlib", "scikit-learn", "seaborn" ]
RUN [ "mamba", "install", "notebook", "-c", "conda-forge","-y"]


EXPOSE 8050

ENTRYPOINT ["python", "dashboardv3.py"]