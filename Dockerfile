FROM python:3.9
EXPOSE 8501
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD streamlit run app.py \
    --server.headless true \
    --browser.serverAddress="0.0.0.0" \
    --server.enableCORS false \
    --browser.gatherUsageStats false
