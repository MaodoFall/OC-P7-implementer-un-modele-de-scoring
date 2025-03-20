# Use the official Python image
FROM python:3.12

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire project to the container
COPY . /code

# Expose the application port (80 for production)
EXPOSE 80

# Command to run the FastAPI app
CMD ["uvicorn", "rest_api.main:app", "--host", "0.0.0.0", "--port", "80"]