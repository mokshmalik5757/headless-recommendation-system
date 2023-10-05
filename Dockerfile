# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install Git
RUN apt-get update && apt-get install -y git

# Set the working directory in the container
WORKDIR /app_docker

# Copy the requirements file into the container at /app
COPY requirements.txt /app_docker/

# Install any needed packages specified in requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app_docker

# Expose the port that your FastAPI application will run on
EXPOSE 8000

# Define the command to run your FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
