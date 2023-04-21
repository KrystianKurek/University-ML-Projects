#!/bin/bash

set -e

./flow_setup.sh &
${NIFI_HOME}/../scripts/start.sh
