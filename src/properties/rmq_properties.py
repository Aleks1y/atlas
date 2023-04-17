import pika

INPUT_QUEUE = "regression_queue_input"
OUTPUT_QUEUE = "regression_queue_output"

credentials = pika.PlainCredentials('rmuser', 'rmpassword')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmqServer', port='5672',
                                                               credentials=credentials))
rmq_channel = connection.channel()
# rmq_parameters = pika.URLParameters('amqp://rmuser:rmpassword@rabbitmqServer:5672')
# rmq_connection = pika.BlockingConnection(rmq_parameters)
# rmq_channel = rmq_connection.channel()
