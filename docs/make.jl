using MCIntegration
using Documenter

DocMeta.setdocmeta!(MCIntegration, :DocTestSetup, :(using MCIntegration); recursive=true)

makedocs(;
    modules=[MCIntegration],
    authors="Kun Chen, Xiansheng Cai",
    repo="https://github.com/numericaleft/MCIntegration.jl/blob/{commit}{path}#{line}",
    sitename="MCIntegration.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://numericaleft.github.io/MCIntegration.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => Any[
            "man/important_sampling.md"
        ],
        "Library" => Any[
                map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
                # "Internals" => map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
        ]
    ],
)

deploydocs(;
    repo="github.com/kunyuan/MCIntegration.jl",
)

# using Lehmann
# using Documenter

# DocMeta.setdocmeta!(Lehmann, :DocTestSetup, :(using Lehmann); recursive=true)

# makedocs(;
#     modules=[Lehmann],
#     authors="Kun Chen, Tao Wang, Xiansheng Cai",
#     repo="https://github.com/kunyuan/Lehmann.jl/blob/{commit}{path}#{line}",
#     sitename="Lehmann.jl",
#     format=Documenter.HTML(;
#         prettyurls=get(ENV, "CI", "false") == "true",
#         canonical="https://kunyuan.github.io/Lehmann.jl",
#         assets=String[],
#     ),
#     pages=[
#         "Home" => "index.md",
#         "Manual" => Any[
#         ],
#         "Library" => Any[
#                 map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
#                 # "Internals" => map(s -> "lib/$(s)", sort(readdir(joinpath(@__DIR__, "src/lib"))))
#         ]
#     ],
# )

# deploydocs(;
#     repo="github.com/kunyuan/Lehmann.jl.git",
#     devbranch="main"
# )