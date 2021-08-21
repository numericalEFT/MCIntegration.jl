using MCIntegration
using Documenter

DocMeta.setdocmeta!(MCIntegration, :DocTestSetup, :(using MCIntegration); recursive=true)

makedocs(;
    modules=[MCIntegration],
    authors="Kun Chen, Xiansheng Cai",
    repo="https://github.com/kunyuan/MCIntegration.jl/blob/{commit}{path}#{line}",
    sitename="MCIntegration.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kunyuan.github.io/MCIntegration.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kunyuan/MCIntegration.jl",
)
