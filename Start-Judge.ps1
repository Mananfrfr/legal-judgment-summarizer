param([int]$Port=8089, [switch]$Install)
$ErrorActionPreference = 'Stop'
$judgeRoot = Join-Path $PSScriptRoot '.runtime\judge'
$model = Join-Path $judgeRoot 'qwen.gguf'
$server = Join-Path $judgeRoot 'bin\llama-server.exe'
function Get-VerifiedAsset($Uri, $Path, $Expected) {
    if (-not (Test-Path -LiteralPath $Path)) {
        $partial = "$Path.partial"
        & curl.exe --fail --location --retry 3 --continue-at - --output $partial $Uri
        if ($LASTEXITCODE -ne 0) { throw "Download interrupted. Run the install command again." }
        if ((Get-FileHash -LiteralPath $partial).Hash -ne $Expected) { throw "Checksum failed for $partial." }
        Move-Item -LiteralPath $partial -Destination $Path
    } elseif ((Get-FileHash -LiteralPath $Path).Hash -ne $Expected) {
        throw "Checksum failed for $Path."
    }
}
if ($Install) {
    New-Item -ItemType Directory -Force $judgeRoot | Out-Null
    $zip = Join-Path $judgeRoot 'cpu.zip'
    Get-VerifiedAsset 'https://github.com/ggml-org/llama.cpp/releases/download/b12445/llama-b12445-bin-win-cpu-x64.zip' $zip '2a224215f248cd32eab5f79a0478ed36492dfeb0c05ac85a82884d6afb386ffc'
    Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $judgeRoot 'bin') -Force
    Get-VerifiedAsset 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true' $model '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
}
if (-not (Test-Path -LiteralPath $model) -or -not (Test-Path -LiteralPath $server)) {
    throw 'Run .\Start-Judge.ps1 -Install once to download the pinned model and Windows CPU runtime.'
}
& $server -m $model --alias qwen2.5-1.5b-instruct-q4_k_m --host 127.0.0.1 --port $Port --ctx-size 4096 --parallel 1 --threads 4 --cors-origins localhost --no-webui --no-slots
exit $LASTEXITCODE
