'''

High level idea: before nccl initialization we need a way to broadcast 
ncclGetUniqueId() to all the ranks 

we can do this via a simple kv store exposed via tcp -- this also means that 
whenevr we do dist.new_group() we can just create a new entry in the store 
rather than doing some ad hoc cpu communication each time 

'''
import socketserver 
# coordination tcp store service surrounding NCCL 
import threading 

class Store: 
    def __init__(self): 
        self.values = {} 
        # lock for server 
        self.condition = threading.Condition()


    # insert or overwrite byte value 
    def set(self, key : str, value: bytes) -> None: 
        pass 

    # block until key exists or timeout expires 
    def get(self, key : str, timeout_s: float = None) -> bytes: 
        pass 

    # atomically increment ingeger value and return new value 
    def add(self, key, delta) -> int: 
        pass 

    # block until every requested key exists 
    def wait(self, keys: list[str], timeout_s: float = None) -> None: 
        pass 

class StoreRequestHandler(socketserver.BaseRequestHandler): 
     def handle(self):
        # Receive data from the client
        data = self.request.recv(1024).strip()
        print(f"Client connected from: {self.client_address}")

        with self.server.store.condition: 
            # process operations 
            pass 


class TCPStoreServer: 
    def __init__(self, host, port): 
        self.store = Store() 
        self.host = host 
        self.port = port 
        self.handler = StoreRequestHandler 

    def start(self): 
        self.server = socketserver.ThreadingTCPServer((self.host, self.port), self.handler) 
        self.server.store = self.store 
        self._server_thread = threading.Thread(target=self.server.serve_forever)
        
        
        self._server_thread.daemon = True
        
    
        self._server_thread.start()
        print(f"tcp store serer started in background on {self.host}:{self.port}")




class TCPStoreClient: 
    pass 

