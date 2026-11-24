# Download American Stories year files (public domain, Hugging Face) one at a time.
# Usage: powershell -File scripts\download_years.ps1 1929 1938 1941 1918
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Years)
$base = 'https://huggingface.co/datasets/dell-research-harvard/AmericanStories/resolve/main/'
$dir = Join-Path $PSScriptRoot '..\data\raw'
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($y in $Years) {
    $out = Join-Path $dir "faro_$y.tar.gz"
    $url = $base + "faro_$y.tar.gz"
    Write-Output "$(Get-Date -Format HH:mm:ss) start $y"
    & curl.exe -s -f -L --retry 10 --retry-delay 5 -C - -o $out $url
    Write-Output "$(Get-Date -Format HH:mm:ss) done $y exit=$LASTEXITCODE size=$((Get-Item $out).Length)"
}
