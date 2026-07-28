from minidist.store import TCPStoreServer, TCPStoreClient

server = TCPStoreServer("0.0.0.0", 12345, timeout_s=10.0)
server.start()

client = TCPStoreClient("0.0.0.0", 12345, timeout_s=10.0)
client.set("test", b"hello")


client2 = TCPStoreClient("0.0.0.0", 12345, timeout_s=10.0)
print(client2.get("test", timeout_s=10.0))


client.close()
client2.close()
server.stop()