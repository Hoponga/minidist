'''

High level idea: before nccl initialization we need a way to broadcast 
ncclGetUniqueId() to all the ranks 

we can do this via a simple kv store exposed via tcp -- this also means that 
whenevr we do dist.new_group() we can just create a new entry in the store 
rather than doing some ad hoc cpu communication each time 

'''
import socketserver 
import socket
# coordination tcp store service surrounding NCCL 
import threading 
import time
import base64


DEFAULT_TIMEOUT_S = 300.0


class Store: 
    def __init__(self): 
        self.values = {} 
        # lock for server 
        self.condition = threading.Condition()


    # insert or overwrite byte value 
    def set(self, key : str, value: bytes) -> None: 
        with self.condition: 
            self.values[key] = value
            self.condition.notify_all() 

    # block until key exists or timeout expires 
    def get(self, key : str, timeout_s: float = DEFAULT_TIMEOUT_S) -> bytes: 
        deadline = time.monotonic() + timeout_s 
        with self.condition: 
            
            while key not in self.values: 
                remaining = deadline - time.monotonic() 
                if remaining <= 0: 
                    return None # timeout 
                self.condition.wait(remaining)
            return self.values[key]

    # atomically increment ingeger value and return new value 
    def add(self, key, delta) -> int: 
        with self.condition: 
            if key not in self.values: 
                self.values[key] = 0
            self.values[key] += delta
            self.condition.notify_all()
            return self.values[key]

    # block until every requested key exists 
    def wait(self, keys: list[str], timeout_s: float = DEFAULT_TIMEOUT_S) -> None: 
        deadline = time.monotonic() + timeout_s 
        with self.condition: 
            while not all(key in self.values for key in keys): 
                remaining = deadline - time.monotonic() 
                if remaining <= 0: 
                    return None # timeout 
                self.condition.wait(remaining)
            return True 

class StoreRequestHandler(socketserver.StreamRequestHandler): 
    def setup(self): 
        super().setup() 
        self.request.settimeout(self.server.timeout_s)

    def handle(self):
        while True: 
            try: 
                line = self.rfile.readline().strip().decode() 
            except socket.timeout: 
                return 
            
            if not line: 
                return 
            
            command = line.split() 
            print(f"Received command: {command}")
            self.handle_command(command) 

    def handle_command(self, cmd_list: list[str]): 
        response = "OK"
        if cmd_list[0] == "set":
            key = cmd_list[1]
            value = base64.b64decode(cmd_list[2])
            self.server.store.set(key, value)
            
        elif cmd_list[0] == "get":
            key = cmd_list[1]
            timeout_s = float(cmd_list[2]) if len(cmd_list) > 2 else self.server.timeout_s

            value = self.server.store.get(key, timeout_s=timeout_s)
            response += f" {base64.b64encode(value).decode()}"
            
        elif cmd_list[0] == "add":
            key = cmd_list[1]
            delta = int(cmd_list[2])
            response += f" {self.server.store.add(key, delta)}"
        elif cmd_list[0] == "wait":
            timeout_s = float(cmd_list[1])
            keys = cmd_list[2:]
            response += f" {self.server.store.wait(keys, timeout_s=timeout_s)}"
        else:
            raise RuntimeError(f"Unknown command: {cmd_list}")
        
        self.wfile.write(response.encode() + b"\n")




class TCPStoreServer: 
    def __init__(self, host, port, timeout_s: float = DEFAULT_TIMEOUT_S): 
        self.store = Store() 
        self.host = host 
        self.port = port 
        self.handler = StoreRequestHandler 
        self.timeout_s = timeout_s

    def start(self): 
        self.server = socketserver.ThreadingTCPServer((self.host, self.port), self.handler, bind_and_activate=False) 
        self.server.store = self.store 
        self.server.timeout_s = self.timeout_s
        self.server.allow_reuse_address = True
        self.server.daemon_threads = True
        
        self.server.server_bind() 
        self.server.server_activate()
        self._server_thread = threading.Thread(target=self.server.serve_forever)
        
        
        self._server_thread.daemon = True
        
    
        self._server_thread.start()
        print(f"tcp store serer started in background on {self.host}:{self.port}")


    def stop(self) -> None: 
        self.server.shutdown() 
        self.server.server_close() 
        self._server_thread.join()




class TCPStoreClient: 
    def __init__(self, host, port, timeout_s: float = DEFAULT_TIMEOUT_S): 
        self.host = host 
        self.port = port 
        self.timeout_s = timeout_s

        self.client_socket = None
        deadline = time.monotonic() + timeout_s

        # hacky way to make sure client doesnt bunk if server is not yet ready can
        delay = 0.05
        while True:
            try:
                self.client_socket = socket.create_connection((self.host, self.port), timeout=min(1.0, max(deadline - time.monotonic(), 0.001)))
                break
            except OSError as err:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"connect to {self.host}:{self.port}") from err
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, 0.5)
        self.client_socket.settimeout(self.timeout_s)    

    
    # send "set <key> <value>\n" to server and wait for response
    def set(self, key : str, value: bytes) -> None: 
        message = f"set {key} {base64.b64encode(value).decode()}\n"
        self.client_socket.sendall(message.encode())
        response = self.client_socket.recv(1024).decode().strip()

        ack = response
        if ack != "OK":
            raise RuntimeError(f"Failed to set key {key}: {response}")

    # send "get <key>\n" to server and wait for response
    def get(self, key : str, timeout_s: float | None = None) -> bytes:
        timeout_s = self.timeout_s if timeout_s is None else timeout_s
        message = f"get {key} {timeout_s}\n"
        self.client_socket.sendall(message.encode())
        response = self.client_socket.recv(1024).decode().strip()
        ack, value = response.split()
        if ack != "OK":
            raise RuntimeError(f"Failed to get key {key}: {response}")
        return base64.b64decode(value)
    
    # send "add <key> <delta>\n" to server and wait for response
    def add(self, key : str, delta: int) -> int: 
        message = f"add {key} {delta}\n"
        self.client_socket.sendall(message.encode())
        response = self.client_socket.recv(1024).decode().strip()
        ack, value = response.split()
        if ack != "OK":
            raise RuntimeError(f"Failed to add key {key}: {response}")
        return int(value)
    
    # send "wait <keys>\n" to server and wait for response
    def wait(self, keys: list[str], timeout_s: float | None = None) -> None:
        timeout_s = self.timeout_s if timeout_s is None else timeout_s
        message = f"wait {timeout_s} {' '.join(keys)}\n"
        self.client_socket.sendall(message.encode())
        response = self.client_socket.recv(1024).decode().strip()
        ack, value = response.split()
        if ack != "OK":
            raise RuntimeError(f"Failed to wait for keys {keys}: {response}")
        return True 
    
    def close(self) -> None: 
        self.client_socket.close()
        self.client_socket = None
        self.host = None
        self.port = None
        self.timeout_s = None
        self.store = None
        self.handler = None
        self.server = None
        self._server_thread = None



