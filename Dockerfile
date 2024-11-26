# Use the official Miniconda3 image as the base image
FROM continuumio/miniconda3:latest

# Set environment variables to prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml file to the container
COPY environment.yml .

# Create the Conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Make sure the environment is activated:
# - Update PATH environment variable to include the Conda environment's binaries
# - This allows us to run commands using the environment's Python and installed packages
ENV PATH /opt/conda/envs/cruelty-free-cosmetics/bin:$PATH

# Copy the rest of the application code to the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Specify the default command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
