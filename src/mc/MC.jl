module MC

import ..Result
import ..report
import ..Configuration
import ..MPIreduceConfig!
import ..addConfig!
import ..clearStatistics!
import ..TINY

using ..Dist
import ..Variable

using ..MCUtility

include("vegas.jl")

end