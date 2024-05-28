FROM python:3.10-bullseye

EXPOSE 8501

RUN pip install streamlit numpy pandas matplotlib
