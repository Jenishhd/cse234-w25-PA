from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        assert src_array.size == dest_array.size, "src and dest arrays must be the same size"
        size = self.Get_size()
        rank = self.Get_rank()
        src_byte = src_array.itemsize * src_array.size
        dest_byte = dest_array.itemsize * dest_array.size  # Equal to src_byte due to assert

        # Update byte count based on role
        if rank == 0:
            # Root receives (size-1) src_bytes and sends (size-1) dest_bytes
            self.total_bytes_transferred += (size - 1) * (src_byte + dest_byte)
        else:
            # Non-root sends one src_byte and receives one dest_byte
            self.total_bytes_transferred += src_byte + dest_byte

        # Step 1: Reduce all data to root (rank 0) using the specified operation
        self.comm.Reduce(src_array, dest_array, op=op, root=0)

        # Step 2: Broadcast the result from root to all processes
        self.comm.Bcast(dest_array, root=0)

               
    def myAlltoall(self, src_array, dest_array):
        nprocs = self.Get_size()
        rank = self.Get_rank()

        # Validate array dimensions
        assert src_array.size % nprocs == 0, "src_array size must be divisible by nprocs"
        assert dest_array.size % nprocs == 0, "dest_array size must be divisible by nprocs"

        # Calculate segment sizes
        seg_size = src_array.size // nprocs
        seg_bytes = src_array.itemsize * seg_size

        # Update total bytes transferred
        self.total_bytes_transferred += 2 * seg_bytes * (nprocs - 1)

        # Prepare send and receive requests arrays
        send_requests = []
        recv_requests = []

        # Process local data first (no communication needed)
        local_send_idx = rank * seg_size
        local_recv_idx = rank * seg_size
        np.copyto(
            dest_array[local_recv_idx:local_recv_idx + seg_size],
            src_array[local_send_idx:local_send_idx + seg_size]
        )

        # Post non-blocking receives first (to avoid potential deadlock)
        for i in range(nprocs):
            if i != rank:  # Skip self
                recv_idx = i * seg_size
                recv_buf = dest_array[recv_idx:recv_idx + seg_size]
                req = self.comm.Irecv(recv_buf, source=i)
                recv_requests.append(req)

        # Then post non-blocking sends
        for i in range(nprocs):
            if i != rank:  # Skip self
                send_idx = i * seg_size
                send_buf = src_array[send_idx:send_idx + seg_size]
                req = self.comm.Isend(send_buf, dest=i)
                send_requests.append(req)

        # Wait for all communications to complete
        MPI.Request.Waitall(send_requests)
        MPI.Request.Waitall(recv_requests)