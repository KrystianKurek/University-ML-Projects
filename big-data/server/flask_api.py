from flask import Flask, jsonify, make_response
from cassandra.cluster import ExecutionProfile, EXEC_PROFILE_DEFAULT, Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import WhiteListRoundRobinPolicy
from cassandra.query import tuple_factory
from flask_cors import CORS

hosts = ['cassandra']
auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')

app = Flask(__name__)
CORS(app)

@app.route('/predictions', methods = ['GET'])
def prediction():
    cluster = Cluster(hosts, auth_provider=auth_provider, port=9042)
    session = cluster.connect()
    row = session.execute("SELECT * FROM apache_air.realtime_views;").all()
    print(row)
    if row:
        return jsonify(row)

@app.route('/insert', methods = ['POST'])
def insertion():
    cluster = Cluster(hosts, auth_provider=auth_provider, port=9042)
    session = cluster.connect()
    session.execute("INSERT INTO apache_air.realtime_views (longtitude, latitude, timestamp, particle, prediction, station_name) VALUES (21.01, 52.23, '2022-01-15 04:05', 'O3', 0.5, 'Warsaw');")
    data = {'message': 'Done', 'code': 'SUCCESS'}
    return make_response(jsonify(data), 201)

@app.route('/batch_views', methods = ['GET'])
def batch_views():
    cluster = Cluster(hosts, auth_provider=auth_provider, port=9042)
    session = cluster.connect()
    rows = session.execute("SELECT * FROM apache_air.batch_views;").all()
    #max_timestamp = max([row.timestamp for row in rows])
    #rows = [row for row in rows if row.timestamp == max_timestamp]

    if rows:
        return jsonify(rows)

app.run(debug=True, host='0.0.0.0', port=1337)