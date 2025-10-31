# Parallelization {#parallelization}

From [the introduction to the domain partitioning](#shell) we know that the domain is partitioned into 10 diamonds, 
that are each partitioned into a number of subdomains, that are each partitioned into cells that correspond to the 
elements of the finite element mesh.

Each subdomain is assigned a (globally) unique ID
``` 
subdomain_id = (diamond_id, subdomain_x, subdomain_y, subdomain_r)
```

Each subdomain is assigned to one of the parallel (MPI) processes. When running on multiple GPUs, each (MPI) process
is exactly assigned to one GPU. Each process may carry more than one subdomain.

To organize the process local subdomains, each `subdomain_id` is mapped to a process-local `local_subdomain_id`,
which is just an integer between 0 and the number of subdomains in the process.

#### Example

An example of a domain partitioning is shown in the figure below. Therein you see (from two angles) a domain partitioned
into 8 subdomains per diamond (subdomain refinement level == 1). The diamond refinement level is 4 (thus we have
2^4 == 16 hex cells in each direction per diamond, and therefore 8 hex cells in each direction per subdomain).
The domain is distributed among 16 processes.

Overall:
```
    global_refinement_level = 4      => 2^4 == 16 hex cells in each direction per diamond
                                     => 16^3 == 4096 hex cells per diamond
                                     => 4096 * 10 = 40960 hex cells globally
                                     => 40960 * 2 == 81920 wedges globally
                                     
    subdomain_refinement_level = 1   => 2^1 == 2 subdomains per direction per diamond
                                     => 2^3 == 8 subdomains per diamond
                                     => 8 * 10 == 80 subdomains globally
                                     => 81920 / 80 == 1024 wedges per subdomain
                                     
    num_processes = 16               => 80 / 16 == 5 subdomains per process (assuming uniform distribution)                               
```

Subdomains colored by MPI rank (hiding diamonds 0, 1, and 5) - from 2 angles:
\image html figures/subdomain_ranks_np16_without_diamonds_0_1_5_angle_0.png
\image html figures/subdomain_ranks_np16_without_diamonds_0_1_5_angle_1.png

Subdomains of MPI rank 10 colored by local subdomain ID - from the same 2 angles:
\image html figures/subdomain_ranks_np16_rank_10_local_subdomains_angle_0.png
\image html figures/subdomain_ranks_np16_rank_10_local_subdomains_angle_1.png