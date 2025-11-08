#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Installing Docker..."

    # Update package index and install prerequisites
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

    # Add Docker's official APT repository
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

    # Update package index again and install Docker
    sudo apt-get update
    sudo apt-get install -y docker-ce

    # Start and enable Docker service
    sudo systemctl start docker
    sudo systemctl enable docker

    echo "Docker installed successfully."
else
    echo "Docker is already installed."
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Installing Docker Compose..."

    # Download the latest version of Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")')/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

    # Apply executable permissions to the binary
    sudo chmod +x /usr/local/bin/docker-compose

    echo "Docker Compose installed successfully."
else
    echo "Docker Compose is already installed."
fi

echo "Stopping all containers..."
sudo docker stop $(sudo docker ps -aq) || true

echo "Removing all containers..."
sudo docker rm $(sudo docker ps -aq) || true

echo "Removing all images..."
sudo docker rmi $(sudo docker images -q) -f || true

echo "Pruning Docker system..."
sudo docker system prune -a -f

cd ..
sudo docker-compose up -d
echo "Deployment completed successfully."