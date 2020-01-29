FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

WORKDIR /exp

RUN ["/opt/conda/bin/conda", "install", "-y", "scipy", "pandas", "dask", "distributed", "scikit-learn", "pytest"]

RUN ["ipython", "-c", "print('create default profile')"]
RUN echo "%load_ext autoreload" > /root/.ipython/profile_default/startup/autoreload.ipy
RUN echo "%autoreload 2" >> /root/.ipython/profile_default/startup/autoreload.ipy

CMD ["python", "-m", "http.server", "8000"]
