# `ExaModels` and `KernelAbstractions` are triggers so that ExaModels' backend extensions load while this one
# precompiles. Needs a functional device at precompile time.
module ExaMittelmannAMDGPUExt

import ExaMittelmann, ExaModels, AMDGPU, KernelAbstractions
import PrecompileTools

PrecompileTools.@compile_workload begin
    if AMDGPU.functional()
        ExaMittelmann._precompile_workload(AMDGPU.ROCBackend())
    end
end

end
