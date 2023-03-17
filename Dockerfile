# Base image
FROM konvoulgaris/diastema-spark-base

# Update system
RUN apt update -y
RUN apt upgrade -y
RUN pip install -U pip

# Create and use project directory
WORKDIR /app

# Copy and install requirements
ADD requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
ADD src/ src/
