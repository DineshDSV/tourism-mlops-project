# Use a minimal Python 3.9 base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all files from the current directory into the container
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860
EXPOSE 7860

# Run the Streamlit app on port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
