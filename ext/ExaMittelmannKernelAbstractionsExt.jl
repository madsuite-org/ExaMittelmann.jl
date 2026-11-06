# Loading KernelAbstractions invalidates CPU code in the package's own image; this image
# loads after those methods, so what it stores stays valid.
module ExaMittelmannKernelAbstractionsExt

import ExaMittelmann, ExaModels, KernelAbstractions
import PrecompileTools

PrecompileTools.@compile_workload ExaMittelmann._precompile_workload()

end
