import socket
import time
import numpy as np
import struct

def send_data_chunk(sock, data):
    # Pack uint16_t values into bytes
    # 'H' format specifier is for unsigned short (uint16_t)
    packed_data = struct.pack(f'!{len(data)}H', *data)
    sock.sendall(packed_data)
    print(f"Sent data chunk of {len(data)} uint16_t values ({len(packed_data)} bytes)")

def main():
    # Connect to C server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 5000)
    print(f"Connecting to {server_address[0]}:{server_address[1]}")
    
    try:
        sock.connect(server_address)
        
        # Number of uint16_t values to send in each chunk
        # Each uint16_t is 2 bytes, so 512 values = 1024 bytes
        num_values = 2048*2048  
        
        while True:
            # Generate random uint16_t values (0-65535)
            data = np.random.randint(0, 65536, num_values, dtype=np.uint16)
            
            # Send the data
            send_data_chunk(sock, data)
            
            # Add a small delay between sends
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Sending terminated by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close connection
        sock.close()
        print("Connection closed")

if __name__ == "__main__":
    main()