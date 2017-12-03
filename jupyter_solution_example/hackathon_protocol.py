from __future__ import print_function # for python 2 compatibility
import hashlib, socket, time, sys

MBODYLEN_LEN = 4
CHECKSUM_LEN = 8

LOGIN = 'LOGIN'
HEADER = 'HEADER'
ORDERBOOK = 'ORDERBOOK'
PREDICT_NOW = 'PREDICT_NOW'
VOLATILITY = 'VOLATILITY'
SCORE = 'SCORE'

MESSAGE_FORMAT = "%%0%dd\t%%s\t%%s" % MBODYLEN_LEN

MAX_MESSAGE_LEN = 10000


def get_hex_checksum(value):

    if isinstance(value, (bytes, bytearray)):  # In python2: string and bytes types are same, this condition is True
        return hashlib.md5(value).hexdigest()[:CHECKSUM_LEN]

    if isinstance(value, str):
        return get_hex_checksum(string_to_bytes(value))

    raise TypeError("Invalid type for get_hex_checksum()")


def py3_string_to_bytes(string_value):
    return bytes(string_value, 'utf-8')


def py3_bytes_to_string(bytes_value):
    return bytes_value.decode("utf-8")


def py2_string_to_bytes(value):
    return value


def py2_bytes_to_string(value):
    return str(value)


if sys.version_info.major == 3:
    string_to_bytes = py3_string_to_bytes
    bytes_to_string = py3_bytes_to_string
    DisconnectError = ConnectionResetError
else:
    string_to_bytes = py2_string_to_bytes
    bytes_to_string = py2_bytes_to_string
    DisconnectError = socket.error


def make_raw_message(message_body):

    if isinstance(message_body, (tuple, list)):
        # tuple support
        return make_raw_message('\t'.join((str(x) for x in message_body)))

    return string_to_bytes(MESSAGE_FORMAT % (len(message_body), get_hex_checksum(message_body), message_body))


class SessionImpl(object):
    def __init__(self, sock, run_result = None):
        self.sock = sock
        self.recv_buffer = bytearray()
        self.send_buffer = bytearray()
        self.run_result = run_result
        self.stopped = False
        self.bytes_recv = 0
        self.start_time = time.time()
        self.sock.settimeout(1.0)

    def is_log_enabled(self): return False

    def send_message(self, message_body):
        return self.send_raw_message(make_raw_message(message_body))

    def send_raw_message(self, message_bytes):
        self.log(True, message_bytes)
        self.send_buffer += message_bytes
        return self

    def run(self):
        prefix_len = MBODYLEN_LEN + 1 + CHECKSUM_LEN + 1 # body_len + tab + checksum + tab

        try:
            while True:
                while len(self.send_buffer) > 0:
                    CHUNK_SIZE = 2*1024
                    self.sock.send(self.send_buffer[:CHUNK_SIZE])
                    del self.send_buffer[:CHUNK_SIZE]

                if self.stopped: break;

                try:
                    # wait until any amount of bytes received
                    just_recv = self.sock.recv(1024*1024)
                except socket.timeout:
                    # timeout
                    self.on_socket_timeout()
                    continue

                if not just_recv: break

                self.recv_buffer += just_recv
                self.bytes_recv += len(just_recv)

                #self.log(None, b"Now received %d, total received %d" % (len(just_recv), self.bytes_recv))

                # try read messages from buffer
                while True:
                    if len(self.recv_buffer) < prefix_len:
                        break

                    body_len = int(self.recv_buffer[:MBODYLEN_LEN])

                    if body_len < 0:
                        raise ValueError("Invalid message len (%d)" % body_len)

                    if body_len > MAX_MESSAGE_LEN:
                        raise ValueError("Too big incoming message len (%d)" % body_len)

                    msg_len = prefix_len + body_len

                    if len(self.recv_buffer) < msg_len:
                        break

                    raw_message = self.recv_buffer[:msg_len]
                    self.log(False, raw_message)
                    checksum = raw_message[MBODYLEN_LEN + 1 : MBODYLEN_LEN + 1 + CHECKSUM_LEN]
                    body = raw_message[prefix_len:]
                    if bytes_to_string(checksum) != get_hex_checksum(body):
                        raise ValueError("Checksum error. body: " + bytes_to_string(body[:10000]))

                    self.on_message(bytes_to_string(body))
                    del self.recv_buffer[:msg_len]

        except (DisconnectError, ValueError) as ex:
            print("Disconnected, because", ex)

        print("TCP Session finished")
        self.sock.close()
        return self.run_result

    def log(self, is_send, raw_message):
        # may be overloaded as well
        if self.is_log_enabled():
            send_or_recv = "[SEND]" if is_send else ("[RECV]" if is_send is not None else " "*6)
            print('%.6f' % (time.time() - self.start_time), send_or_recv, bytes_to_string(raw_message))
        return

    def stop(self):
        self.stopped = True

    def on_message(self, message_body):
        pass

    def on_socket_timeout(self):
        pass


class Client(SessionImpl):
    def __init__(self, sock):
        super(Client, self).__init__(sock)

    def send_login(self, username, pass_hash):
        return self.send_message((LOGIN, username, pass_hash))

    def send_volatility(self, volatility):

        if not isinstance(volatility, (float, int)):
            raise ValueError("send_volatility: volatility be float (actual {})".format(type(volatility)))

        return self.send_message((VOLATILITY, volatility))

    def on_header(self, csv_header):
        # should be overridden
        pass

    def on_orderbook(self, cvs_line_values):
        # should be overridden
        pass

    def on_score(self, items_processed, time_elapsed, score_value):
        # should be overridden
        pass

    def make_prediction(self):
        # should be overridden
        pass

    def on_message(self, message):
        tokens = message.split('\t')
        if tokens[0] == ORDERBOOK:
            # 0 = ORDERBOOK
            # 1 = instrument
            # 2 = time
            # 3 = price0
            # 4 = vol0
            # ...
            instrument = tokens[1]
            time_str = tokens[2]
            cvs_line_items = [instrument, time_str] + [float(tokens[n]) for n in range(3, len(tokens))]
            self.on_orderbook(cvs_line_items)

        elif tokens[0] == PREDICT_NOW:
            self.make_prediction()

        elif tokens[0] == HEADER:
            self.on_header(tokens[1:])

        elif tokens[0] == SCORE:
            self.on_score(int(tokens[1]), float(tokens[2]), float(tokens[3]))


def prepare_header_raw_message(cvs_line_values):
    return make_raw_message((HEADER,) + tuple(cvs_line_values))


def prepare_orderbook_raw_message(cvs_line_values):
    return make_raw_message((ORDERBOOK,) + tuple(cvs_line_values))


def prepare_predict_now_raw_message():
    return make_raw_message(PREDICT_NOW)


class Server(SessionImpl):
    def __init__(self, sock, run_result = None):
        super(Server, self).__init__(sock, run_result)

    def send_score(self, items_processed, time_elapsed, score_value):
        return self.send_message((SCORE, items_processed, time_elapsed, score_value))

    def on_login(self, username, pass_hash):
        # should be overridden
        pass

    def on_volatility(self, volatility):
        # should be overridden
        pass

    def on_message(self, message):
        tokens = message.split('\t')

        if tokens[0] == VOLATILITY:
            self.on_volatility(float(tokens[1]))

        if tokens[0] == LOGIN:
            self.on_login(tokens[1], tokens[2])


# helper TCP functions
def tcp_listen(host, port, accept_handler):
    acceptor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    acceptor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    acceptor.bind((host, port))
    acceptor.listen(5)
    while True:
        connection, address = acceptor.accept()
        print('Accepted from', address, '; TCP session started.')
        res = accept_handler(connection, address)
        if res: break
    acceptor.close()


def tcp_connect(ip_address, port, connect_handler):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip_address, port))
    print('Connected to', (ip_address, port), '; TCP session started.')
    connect_handler(sock)
