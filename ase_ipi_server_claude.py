#!/usr/bin/env python3
"""
CHGNet Calculator Server for gRASPA Monte Carlo

Simple server that provides CHGNet energy calculations for gRASPA.

Usage:
    python chgnet_server.py --socket graspa_mc
    python chgnet_server.py --port 31415
"""

import sys
import argparse
from ase import Atoms
from ase.calculators.socketio import SocketIOCalculator
from chgnet.model import CHGNet
from chgnet.model import CHGNetCalculator

def main():
    parser = argparse.ArgumentParser(description='CHGNet server for gRASPA MC')
    
    parser.add_argument('--socket', '-s',
                       help='UNIX socket name (creates /tmp/ipi_<name>)')
    
    parser.add_argument('--port', '-p', type=int,
                       help='INET socket port (alternative to --socket)')
    
    parser.add_argument('--device', '-d', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for CHGNet (default: cuda)')
    
    args = parser.parse_args()
    
    # Validate socket arguments
    if not args.socket and not args.port:
        print("Error: Must specify either --socket or --port")
        sys.exit(1)
    
    if args.socket and args.port:
        print("Error: Cannot specify both --socket and --port")
        sys.exit(1)
    
    print("="*70)
    print("CHGNet Calculator Server for gRASPA Monte Carlo")
    print("="*70)
    
    # Setup CHGNet
    print(f"\nLoading CHGNet on {args.device}...")
    try:
        calc = CHGNetCalculator(use_device=args.device)
        print("CHGNet loaded successfully")
    except ImportError:
        print("Error: CHGNet not installed. Install with:")
        print("  pip install chgnet")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CHGNet: {e}")
        sys.exit(1)
    
    # Setup socket
    socket_kwargs = {'log': sys.stdout}
    
    if args.socket:
        socket_kwargs['unixsocket'] = args.socket
        print(f"\nSocket: /tmp/ipi_{args.socket} (UNIX domain)")
        print(f"\ngRASPA command:")
        print(f"  ./graspa input.dat --ase-socket {args.socket}:UNIX\n")
    else:
        socket_kwargs['port'] = args.port
        print(f"\nSocket: localhost:{args.port} (INET)")
        print(f"\ngRASPA command:")
        print(f"  ./graspa input.dat --ase-socket localhost:{args.port}\n")
    
    print("Waiting for gRASPA to connect...")
    
    # Create dummy atoms (will be updated with real positions from gRASPA)
    atoms = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    
    # Start the socket server
    try:
        with SocketIOCalculator(calc=calc, **socket_kwargs) as io_calc:
            atoms.calc = io_calc
            
            print("Server started. Processing energy requests from gRASPA...")
            print("Press Ctrl+C to stop.\n")
            
            # Keep server alive
            import time
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()