#!/bin/bash

while ! cqlsh cassandra -e 'describe tables;' ; do
    sleep 5
done

echo "Creating keyspace and tables..."
cqlsh cassandra -f /schema.cql
