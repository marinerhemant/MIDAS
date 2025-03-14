import socket
import struct
import sys

def receive_doubles():
    """
    Receive double values with a two-part uint16 header:
    - First uint16: frame number
    - Second uint16: count of doubles
    Returns a tuple of (frame_num, values) or (None, None) on failure
    """
    # Define server address and port
    HOST = '127.0.0.1'
    PORT = 5002
    
    # Create socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # Bind socket to address and port
        server_socket.bind((HOST, PORT))
        # Listen for connections
        server_socket.listen(1)
        print(f"Server listening on {HOST}:{PORT}")
        
        # Accept client connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")
        
        # Receive dataset number header (first uint16)
        dataset_header = client_socket.recv(2)  # uint16 = 2 bytes
        if not dataset_header or len(dataset_header) != 2:
            print("Failed to receive dataset number header")
            return None, None, None
        
        # Unpack dataset number: '!' for network byte order (big-endian), 'H' for uint16
        dataset_num = struct.unpack('!H', dataset_header)[0]
        
        # Receive count header (second uint16)
        count_header = client_socket.recv(2)  # uint16 = 2 bytes
        if not count_header or len(count_header) != 2:
            print("Failed to receive count header")
            return None, None
        
        # Unpack count: '!' for network byte order (big-endian), 'H' for uint16
        n_doubles = struct.unpack('!H', count_header)[0]
        
        print(f"Receiving frame {frame_num} with {n_doubles} doubles")
        
        # Receive double values
        values = []
        for i in range(n_doubles):
            # Receive each double (8 bytes)
            double_data = client_socket.recv(8)  # double = 8 bytes
            if not double_data or len(double_data) != 8:
                print(f"Failed to receive value {i+1}")
                return None, None
            
            # Unpack double: '!' for network byte order, 'd' for double
            # For portability, we unpack as a 64-bit integer and then convert to double
            network_value = struct.unpack('!Q', double_data)[0]  # 'Q' for uint64
            double_value = struct.unpack('d', struct.pack('Q', network_value))[0]
            values.append(double_value)
            
        print(f"Received frame {frame_num} with {len(values)} doubles successfully")
        
        return frame_num, values
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        server_socket.close()

if __name__ == "__main__":
    # Receive and print the values
    frame_num, values = receive_doubles()
    if values is not None:
        print(f"Received frame {frame_num}:")
        for i, value in enumerate(values):
            print(f"  [{i}]: {value}")
    else:
        print("Failed to receive values")
        sys.exit(1)