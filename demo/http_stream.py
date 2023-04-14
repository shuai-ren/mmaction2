from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from threading import Condition, Thread
import json
import cv2

class OutPut(object):
    def __init__(self):
        self.data = None
        self.condition = Condition()

    def write(self, data):
        with self.condition:
            self.data = data
            self.condition.notify_all()


def socket_send(q_url, port):
    class StreamServer(socketserver.ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    class SteamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/json':
                # print('=========== json ==================')
                self.send_response(200)
                # self.send_header("Content-type", "text/html;charset=%s" % "UTF-8")
                self.send_header('Accept-Range', 'bytes')
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header("Content-type", "application/json")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                try:
                    while True:
                        with json_out.condition:
                            json_out.condition.wait()
                            json_data = json_out.data
                            json_data = json.dumps(json_data).encode("UTF-8")
                            self.wfile.write(json_data)
                            self.wfile.write(b'\r\n')
                except Exception as e:
                    print(e)

            elif self.path == '/jpeg':
                # print('=========== mjpeg ==================')
                self.send_response(200)
                self.send_header('Age', 0)
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header(
                    'Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                self.end_headers()

                try:
                    while True:
                        with jpeg_out.condition:
                            jpeg_out.condition.wait()
                            frame = jpeg_out.data
                            jpg_str = cv2.imencode('.jpg', frame)[1].tostring()

                            self.wfile.write(b'--FRAME\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', len(jpg_str))
                            self.end_headers()
                            self.wfile.write(jpg_str)
                            self.wfile.write(b'\r\n')

                except Exception as e:
                    print(e)

    class GetData():
        def __init__(self, q_url, json_out, jpeg_out):
            self.q_url = q_url
            self.json_out = json_out
            self.jpeg_out = jpeg_out

        def __enter__(self):
            self.thread = Thread(target=self.process)
            self.thread.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.thread.join()

        def process(self):
            while True:
                try:
                    data = q_url.get()
                    self.json_out.write(data[0])
                    self.jpeg_out.write(data[1])
                except Exception as e:
                    print(e)
                    break

    json_out = OutPut()
    jpeg_out = OutPut()

    with GetData(q_url, json_out, jpeg_out) as get:
        port_address = ('', int(port))
        http_server = StreamServer(port_address, SteamHandler)
        http_server.serve_forever()

def start_socket_send(q_url, http_port):
    thread_read = Thread(target=socket_send, args=(q_url, http_port), daemon=True)
    thread_read.start()
