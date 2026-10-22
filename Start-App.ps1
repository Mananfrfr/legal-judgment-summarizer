param([string]$Encoder='', [string]$Bundle='', [int]$Port=7861)
$ErrorActionPreference='Stop'
$appRoot=$PSScriptRoot
$assetRoot=Join-Path (Split-Path $appRoot -Parent) 'stage4_upgrade'
$localPython=Join-Path $appRoot '.venv\Scripts\python.exe'
$workspacePython=Join-Path $assetRoot '.verify-env\Scripts\python.exe'
if(Test-Path -LiteralPath $localPython){
    $appPython=$localPython
    $env:PYTHONPATH=Join-Path $appRoot 'src'
}elseif(Test-Path -LiteralPath $workspacePython){
    $appPython=$workspacePython
    $env:PYTHONPATH=@((Join-Path $appRoot 'src'),(Join-Path $assetRoot 'test_dependencies'),(Join-Path (Split-Path $appRoot -Parent) '.test-deps')) -join ';'
}else{throw 'Install the isolated .venv first; see README.md.'}
if(-not $Encoder){
    $Encoder=Join-Path $appRoot 'models\encoder'
    if(-not (Test-Path -LiteralPath $Encoder)){$Encoder=Join-Path $assetRoot 'models\encoder'}
}
if(-not $Bundle){
    $Bundle=Join-Path $appRoot 'models\stage4_english_model_bundle.zip'
    if(-not (Test-Path -LiteralPath $Bundle)){$Bundle=Join-Path $assetRoot 'models\stage4_english_model_bundle.zip'}
}
if(-not (Test-Path -LiteralPath $Encoder) -or -not (Test-Path -LiteralPath $Bundle)){
    throw 'Model files are missing. Run python scripts/download_models.py or pass -Encoder and -Bundle.'
}
$env:LEGAL_ENCODER_DIR=(Resolve-Path -LiteralPath $Encoder).Path
$env:LEGAL_MODEL_BUNDLE=(Resolve-Path -LiteralPath $Bundle).Path
$env:LEGAL_REVIEW_DATA=Join-Path $appRoot '.runtime'
$env:HF_HUB_OFFLINE='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:LEGAL_CPU_THREADS='4'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:APP_HOST='127.0.0.1'
$env:APP_PORT=[string]$Port
Write-Host "Open http://127.0.0.1:$Port. Keep this window open; Ctrl+C stops the server."
& $appPython (Join-Path $appRoot 'src\stage44_app.py')
exit $LASTEXITCODE
