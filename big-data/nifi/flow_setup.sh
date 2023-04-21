#!/bin/bash

until [ -f /opt/nifi/nifi-current/logs/nifi-app.log ]
do
     sleep 5
done

tail -f /opt/nifi/nifi-current/logs/nifi-app.log | sed '/Started Application Controller/ q'

TOKEN=`curl "https://$(hostname):8443/nifi-api/access/token" -k -H 'Content-Type: application/x-www-form-urlencoded; charset=UTF-8' --data 'username=admin&password=adminadminadmin' --compressed`

REGISTRY_ID=`curl "https://$(hostname):8443/nifi-api/controller/registry-clients/" -k -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -d '{"component": {"name": "dd", "properties": {"url": "http://nifi-registry:18080"}, "type": "org.apache.nifi.registry.flow.NifiRegistryFlowRegistryClient"}, "revision": {"version": 0}}' | jq .id`

VERSION_CONTROL="
	\"versionControlInformation\": {
		\"bucketId\": \"eee03c57-7e3c-4bba-9818-f8f9ba231bfd\",
		\"flowId\": \"0dcae8cd-d4ad-42d5-a666-84fcd4886551\",
		\"registryId\": $REGISTRY_ID,
		\"version\": 15
	}"

curl "https://$(hostname):8443/nifi-api/process-groups/root/process-groups" -k -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -d "{\"component\": {$VERSION_CONTROL}, \"revision\": {\"version\": 0}}"
