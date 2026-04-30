# Custom Image Template 

 ```ghcr.io/marcpezz/volcan:main``` 


julia --project=. -e 'using Pkg; Pkg.add(["JSON", "HDF5"])'
julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. julia/run_all_scenarios.jl
