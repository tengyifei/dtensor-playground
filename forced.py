import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate, Partial
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.functional import gelu

def setup_distributed():
    """Initialize distributed environment"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{rank}")
        )
    
    return rank, world_size

def force_jax_pattern_matmul1(input_tensor, w_in, mesh_2d, rank):
    """
    Force the JAX pattern for first matmul: In[B_X, D_Y] @ W_in[D_X, F_Y]
    Should produce Intermediate[B_X, F_Y]
    """
    # Manually redistribute to ensure correct placements for JAX pattern
    # Input needs D dimension replicated: [B_X, D_Y] -> [B_X, D_replicated]
    input_for_matmul = input_tensor.redistribute(
        placements=[Shard(0), Replicate()]
    )
    
    # W_in needs D dimension replicated: [D_X, F_Y] -> [D_replicated, F_Y]
    w_in_for_matmul = w_in.redistribute(
        placements=[Replicate(), Shard(1)]
    )
    
    if rank == 0:
        print(f"After redistribution for matmul1:")
        print(f"  Input: {input_for_matmul.placements}")
        print(f"  W_in: {w_in_for_matmul.placements}")
    
    # Now matmul should produce [B_X, F_Y] directly
    result = torch.matmul(input_for_matmul, w_in_for_matmul)
    
    if rank == 0:
        print(f"  Result: {result.placements}")
    
    return result

def force_jax_pattern_matmul2(intermediate, w_out, mesh_2d, rank):
    """
    Force the JAX pattern for second matmul: Intermediate[B_X, F_Y] @ W_out[F_Y, D_X]
    Should do local matmul -> Partial -> reduce-scatter to get Output[B_X, D_Y]
    """
    # For JAX pattern, we want F dimensions to match
    # Intermediate is [B_X, F_Y], W_out is [F_Y, D_X]
    # F dimensions already match, so local matmul should work
    
    if rank == 0:
        print(f"Before matmul2:")
        print(f"  Intermediate: {intermediate.placements}")
        print(f"  W_out: {w_out.placements}")
    
    # Do the matmul - should produce partial results
    result = torch.matmul(intermediate, w_out)
    
    if rank == 0:
        print(f"  Raw result: {result.placements}")
    
    # Check if we got the expected pattern
    expected_pattern = (Shard(0), Partial())  # B_X, D_partial
    if result.placements != expected_pattern:
        if rank == 0:
            print(f"  WARNING: Got {result.placements}, expected {expected_pattern}")
    
    # Force redistribution to final pattern [B_X, D_Y]
    final_result = result.redistribute(
        placements=[Shard(0), Shard(1)]
    )
    
    if rank == 0:
        print(f"  Final result: {final_result.placements}")
    
    return final_result

def main():
    rank, world_size = setup_distributed()
    
    if world_size != 8:
        if rank == 0:
            print(f"This example requires exactly 8 GPUs, but got {world_size}")
        if world_size > 1:
            dist.destroy_process_group()
        return
    
    # Initialize 2D mesh: 4x2 (4 for FSDP, 2 for TP)
    mesh_2d = init_device_mesh("cuda", (4, 2), mesh_dim_names=("fsdp", "tp"))
    
    # Model dimensions
    B = 1024  # Total batch size
    D = 2048  # Hidden dimension (d_model)
    F = 16384 # Feed-forward dimension (d_ff)
    
    # Create tensors with exact JAX sharding pattern
    # Input: In[B_X, D_Y]
    input_placement = [Shard(0), Shard(1)]
    input_local = torch.randn(B // 4, D // 2, device=f"cuda:{rank}")
    input_tensor = DTensor.from_local(
        input_local,
        device_mesh=mesh_2d,
        placements=input_placement
    )
    
    # W_in: [D_X, F_Y]
    w_in_placement = [Shard(0), Shard(1)]
    w_in_local = torch.randn(D // 4, F // 2, device=f"cuda:{rank}")
    w_in = DTensor.from_local(
        w_in_local,
        device_mesh=mesh_2d,
        placements=w_in_placement
    )
    
    # W_out: [F_Y, D_X]
    w_out_placement = [Shard(1), Shard(0)]
    w_out_local = torch.randn(F // 2, D // 4, device=f"cuda:{rank}")
    w_out = DTensor.from_local(
        w_out_local,
        device_mesh=mesh_2d,
        placements=w_out_placement
    )
    
    if rank == 0:
        print("="*60)
        print("FORCING JAX PATTERN")
        print("="*60)
    
    dist.barrier(device_ids=[rank])
    
    # First matmul with forced pattern
    if rank == 0:
        print("\nStep 1: Forcing JAX pattern for In @ W_in")
    
    intermediate = force_jax_pattern_matmul1(input_tensor, w_in, mesh_2d, rank)
    
    # Check if we got expected [B_X, F_Y]
    expected = [Shard(0), Shard(1)]
    if intermediate.placements != tuple(expected):
        if rank == 0:
            print(f"WARNING: Intermediate is {intermediate.placements}, expected {expected}")
    
    # Apply activation
    intermediate = gelu(intermediate)
    
    # Second matmul with forced pattern
    if rank == 0:
        print("\nStep 2: Forcing JAX pattern for Intermediate @ W_out")
    
    output = force_jax_pattern_matmul2(intermediate, w_out, mesh_2d, rank)
    
    # Verify final output
    if tuple(output.placements) == tuple(input_placement):
        if rank == 0:
            print(f"\nSUCCESS: Output matches input placement {output.placements}")
    else:
        if rank == 0:
            print(f"\nFAILED: Output {output.placements} doesn't match input {input_placement}")
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    comm_mode = CommDebugMode()
    with comm_mode:
        main()
    
    if int(os.environ.get("RANK", 0)) == 0:
        print("\n" + "="*60)
        print("COMMUNICATION SUMMARY:")
        print("="*60)
        comm_counts = comm_mode.get_comm_counts()
        for op, count in comm_counts.items():
            print(f"{op}: {count}")
        
        # Analyze communication pattern
        has_allreduce = any('all_reduce' in str(op) for op in comm_counts.keys())
        has_allgather = any('all_gather' in str(op) for op in comm_counts.keys())
        has_alltoall = any('alltoall' in str(op) for op in comm_counts.keys())
        has_reducescatter = any('reduce_scatter' in str(op) for op in comm_counts.keys())
        
        print("\nCommunication Analysis:")
        print(f"- All-reduce: {'YES' if has_allreduce else 'NO'}")
        print(f"- All-gather: {'YES' if has_allgather else 'NO'}")
        print(f"- All-to-all: {'YES' if has_alltoall else 'NO'}")
        print(f"- Reduce-scatter: {'YES' if has_reducescatter else 'NO'}")
        
        if has_allreduce:
            print("\nWARNING: All-reduce was used! This violates the JAX pattern requirement.")
