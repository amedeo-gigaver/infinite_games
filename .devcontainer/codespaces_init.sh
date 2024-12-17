#!/bin/bash

(echo && cat ./.devcontainer/aliases.sh) >> ~/.bashrc

git config --global push.autoSetupRemote true
cp ./.devcontainer/pre-commit ./.git/hooks/pre-commit
