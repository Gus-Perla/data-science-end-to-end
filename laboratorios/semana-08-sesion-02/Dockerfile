FROM python:3.12-slim

WORKDIR /app

COPY requierments.txt .

RUN pip install -r requierments.txt

COPY . .

CMD [ "python3", "./starter/EDA.py" ]