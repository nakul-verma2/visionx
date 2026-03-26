# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Create a non-root user and switch to it for Hugging Face compatibility
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy the requirements file into the container
COPY --chown=user requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY --chown=user . .

# Create the uploads folder in /tmp as configured in app.py
RUN mkdir -p /tmp/visionx_uploads && chmod 777 /tmp/visionx_uploads

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# Specify the command to run the application
CMD ["python", "app.py"]
