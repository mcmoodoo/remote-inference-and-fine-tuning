default:
    @just --list

# connect to the pod
connect host:
    ssh {{host}}@ssh.runpod.io -i ~/.ssh/runpod_ed25519

# create a default pod
create-pod:
    runpodctl create pod --gpuType "NVIDIA GeForce RTX 5090" --imageName runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 --volumeSize 50 --name shorts-maker --ports 8888/http,22/tcp --startSSH --env "ACCEPT_EULA=true"
