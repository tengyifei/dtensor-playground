import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate, Partial
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.functional import gelu
import time

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
    
    # Calculate local shard sizes
    local_B = B // 4  # Batch sharded on FSDP dimension
    local_D = D // 2  # Hidden dim sharded on TP dimension
    local_F = F // 2  # Feedforward dim sharded on TP dimension
    
    # Create tensors following JAX pattern exactly
    # Input: In[B_X, D_Y]
    input_placement = [Shard(0), Shard(1)]
    input_local = torch.randn(local_B, local_D, device=f"cuda:{rank}")
    input_tensor = DTensor.from_local(
        input_local,
        device_mesh=mesh_2d,
        placements=input_placement
    )
    
    # W_in: [D_X, F_Y] 
    w_in_placement = [Shard(0), Shard(1)]
    w_in_local = torch.randn(D // 4, local_F, device=f"cuda:{rank}")
    w_in = DTensor.from_local(
        w_in_local,
        device_mesh=mesh_2d,
        placements=w_in_placement
    )
    
    # W_out: [F_Y, D_X] - Keep D sharded on X axis!
    w_out_placement = [Shard(1), Shard(0)]
    w_out_local = torch.randn(local_F, D // 4, device=f"cuda:{rank}")
    w_out = DTensor.from_local(
        w_out_local,
        device_mesh=mesh_2d,
        placements=w_out_placement
    )
    
    if rank == 0:
        print("="*60)
        print("Initial tensor placements:")
        print(f"Input: {input_tensor.placements} on shape {input_tensor.shape}")
        print(f"W_in: {w_in.placements} on shape {w_in.shape}")
        print(f"W_out: {w_out.placements} on shape {w_out.shape}")
        print("="*60)
    
    dist.barrier(device_ids=[rank])
    
    if rank == 0:
        print("\nFORWARD PASS")
        print("="*60)
    
    # First matmul: In[B_X, D_Y] @ W_in[D_X, F_Y]
    # DTensor should automatically handle the all-gathers
    if rank == 0:
        print("\nStep 1: In @ W_in")
        print("Before matmul:")
        print(f"  Input: {input_tensor.placements}")
        print(f"  W_in: {w_in.placements}")
    
    intermediate = torch.matmul(input_tensor, w_in)
    
    if rank == 0:
        print(f"\nIntermediate result: {intermediate.placements} on shape {intermediate.shape}")
    
    # Apply activation
    intermediate = gelu(intermediate)
    
    # Second matmul: Intermediate[B_X, F_Y] @ W_out[F_Y, D_X]
    # This should produce [B_X, D_partial] then reduce-scatter
    if rank == 0:
        print("\nStep 2: Intermediate @ W_out")
        print("Before matmul:")
        print(f"  Intermediate: {intermediate.placements}")
        print(f"  W_out: {w_out.placements}")
    
    # Don't redistribute W_out! Let DTensor handle it
    output = torch.matmul(intermediate, w_out)
    
    if rank == 0:
        print(f"\nOutput after matmul: {output.placements} on shape {output.shape}")
    
    # Now let's check what DTensor produced and potentially fix it
    # The output should ideally be [B_X, D_Y] to match input
    if output.placements != input_placement:
        if rank == 0:
            print(f"\nOutput placement {output.placements} doesn't match input {input_placement}")
            print("Redistributing...")
        
        # Check if we have Partial placement that needs reduction
        has_partial = any(isinstance(p, Partial) for p in output.placements)
        if has_partial and rank == 0:
            print("Output has Partial placement - will be reduced during redistribution")
        
        output = output.redistribute(placements=input_placement)
        
        if rank == 0:
            print(f"Final output: {output.placements} on shape {output.shape}")
    
    time.sleep(2)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    comm_mode = CommDebugMode()
    with comm_mode:
        main()
    
    # Print communication summary
    if int(os.environ.get("RANK", 0)) == 0:
        print("\n" + "="*60)
        print("COMMUNICATION SUMMARY:")
        print("="*60)
        comm_counts = comm_mode.get_comm_counts()
        for op, count in comm_counts.items():
            print(f"{op}: {count}")
        
        # Check if all-reduce was used
        has_allreduce = any('all_reduce' in str(op) for op in comm_counts.keys())
        if has_allreduce:
            print("\nWARNING: All-reduce was used! This doesn't match the JAX pattern.")
        else:
            print("\nGood: No all-reduce operations detected.")
