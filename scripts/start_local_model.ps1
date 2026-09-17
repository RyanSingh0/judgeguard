param([string]$Device = '')
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
$assetDir = Join-Path $projectDir '.models'
New-Item -ItemType Directory -Force $assetDir | Out-Null
$runtimeZip = Join-Path $assetDir 'llama-b11012.zip'
$runtimeDir = Join-Path $assetDir 'llama-b11012'
$modelFile = Join-Path $assetDir 'Qwen3-4B-Q4_K_M.gguf'
$runtimeHash = '0e2f8e22e2019a781db3360d3ccb0b1ff7cffb14dbc9837b54e09f9159878424'
$modelHash = '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
if (-not (Test-Path -LiteralPath $runtimeZip)) {
    Invoke-WebRequest 'https://github.com/ggml-org/llama.cpp/releases/download/b11012/llama-b11012-bin-win-vulkan-x64.zip' -OutFile $runtimeZip
}
if ((Get-FileHash -LiteralPath $runtimeZip -Algorithm SHA256).Hash.ToLower() -ne $runtimeHash) { throw 'Runtime hash mismatch; download a clean copy.' }
Expand-Archive -LiteralPath $runtimeZip -DestinationPath $runtimeDir -Force
if (-not (Test-Path -LiteralPath $modelFile)) {
    Write-Host 'Downloading 2.5 GB model. This is a one-time download.'
    Invoke-WebRequest 'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf' -OutFile $modelFile
}
if ((Get-FileHash -LiteralPath $modelFile -Algorithm SHA256).Hash.ToLower() -ne $modelHash) { throw 'Model hash mismatch; download a clean copy.' }
$serverExe = Join-Path $runtimeDir 'llama-server.exe'
if (-not $Device) {
    $deviceList = & $serverExe --list-devices 2>&1 | Out-String
    if ($deviceList -match '(Vulkan\d+): NVIDIA') { $Device = $Matches[1] }
    else { throw 'No NVIDIA Vulkan device found. Check --list-devices and pass -Device explicitly.' }
}
Write-Host 'Starting local model on http://127.0.0.1:8081. Keep this terminal open; Ctrl+C stops it.'
& $serverExe -m $modelFile --host 127.0.0.1 --port 8081 --device $Device -ngl 99 -c 8192 -np 1 --reasoning off --alias qwen3-4b-q4km-bc640142
