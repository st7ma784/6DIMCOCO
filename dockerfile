# Use the pytorch base image
FROM pytorch/pytorch:latest
RUN apt-get -y update
RUN apt-get -y install git

# Copy the requirements.txt file to the container
COPY requirements.txt /app/requirements.txt
RUN python3 -V
RUN python -V
# Install the Python dependencies
RUN pip install -r /app/requirements.txt

# Set the working directory
WORKDIR /app

# Copy the rest of the files to the container
COPY . /app
RUN python3 -V
RUN python -V
RUN mkdir /data
RUN mkdir /data/artifacts
# Run the training command with the new structure
CMD python scripts/run_training.py --dir /data --wandb
