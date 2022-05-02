FROM python:3.9


WORKDIR /src/


COPY ./requirements.txt .

RUN pip install -U pip setuptools wheel \
    && pip install -r requirements.txt

COPY ./artifacts artifacts/

COPY ./src .

# ENTRYPOINT ["/bin/bash"]

CMD ["./run_app"]