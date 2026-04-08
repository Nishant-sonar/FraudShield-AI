# Use the official Python image
FROM python:3.12-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Update system packages and install dependencies
RUN apt update -y && apt install awscli -y

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit dashboard
CMD ["streamlit", "run", "app_professional_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]