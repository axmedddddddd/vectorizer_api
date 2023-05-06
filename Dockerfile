FROM revolutionsystems/python:3.8-wee-lto-optimized as builder

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip wheel -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

FROM revolutionsystems/python:3.8-wee-lto-optimized

WORKDIR /code

COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

COPY ./pyproject.toml /code/pyproject.toml
COPY ./vectorServer /code/vectorServer

RUN pip install -e .