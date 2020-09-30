FROM pytorch/pytorch:latest

WORKDIR /app

RUN pip install scipy
RUN pip install sklearn
RUN pip install flask

COPY . ./pic2sgf
COPY integration_test_2.py /app/

CMD python integration_test_2.py