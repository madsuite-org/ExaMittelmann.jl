# `ExaModels` and `KernelAbstractions` are triggers so that ExaModels' backend extensions load while this one
# precompiles. Needs a functional device at precompile time.
module ExaMittelmannCUDAExt

import ExaMittelmann, ExaModels, CUDA, KernelAbstractions
import PrecompileTools

PrecompileTools.@compile_workload begin
    if CUDA.functional()
        ExaMittelmann._precompile_workload(CUDA.CUDABackend())
    end
end

end
