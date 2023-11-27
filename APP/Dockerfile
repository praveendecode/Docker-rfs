FROM python

WORKDIR /app

COPY check.py .

COPY RFS_Cleaned_Dataset.csv .

COPY requirements.txt .

RUN pip install -r  requirements.txt 

CMD python check.py