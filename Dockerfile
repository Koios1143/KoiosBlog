FROM ubuntu:jammy
RUN apt update
RUN apt install ca-certificates curl gnupg -y
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
RUN apt-get install -y nodejs vim
RUN apt install build-essential -y
RUN npm install -g @vue/cli
