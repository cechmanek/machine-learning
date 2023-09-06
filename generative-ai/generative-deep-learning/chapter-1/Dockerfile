# Use the official TensorFow image as the base image
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the local directory contents into the container at /app
COPY . /app

# Install additional dependencies
#RUN pip install -r requirements.txt

# Run the script when the container starts
RUN ["python", "my_model.py"]

# build this with:
# docker build -t my-tensorflow-app .

# run with:
# docker run my-tensorflow-app

# WARNING: You are running this container as root, which can cause new files in
# mounted volumes to be created as the root user on your host machine.

# To avoid this, run the container by specifying your user's userid:
# $ docker run -u $(id -u):$(id -g) args...


# Use the docker cp command to copy the file from the container to your local machine.
# The basic syntax is as follows:
# docker cp <container_id>:/path/to/file/on/container /path/to/destination/on/host