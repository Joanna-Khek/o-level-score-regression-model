FROM python:3.9

WORKDIR /opt

RUN pip install --upgrade pip

COPY . /opt
RUN pip install -r requirements.txt
RUN pip install -e .

RUN chmod +x /opt/run.sh
EXPOSE 8001

CMD ["bash", "./run.sh"]