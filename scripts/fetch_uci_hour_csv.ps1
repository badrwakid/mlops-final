# Fetches UCI "Bike Sharing" hour.csv into data/raw/hour.csv when DVC stub says "missing data source".
# Verified against data/raw/hour.csv.dvc (md5 d50bd5a6f55131e72a7bedc334e2fce1).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
# Two-arg Join-Path only (Windows PowerShell 5.1 compatible)
$destDir = Join-Path (Join-Path $root "data") "raw"
$dest = Join-Path $destDir "hour.csv"
New-Item -ItemType Directory -Path $destDir -Force | Out-Null

$zip = Join-Path ([System.IO.Path]::GetTempPath()) "bike-sharing-uci-$([Guid]::NewGuid().ToString('N')).zip"
$expand = Join-Path ([System.IO.Path]::GetTempPath()) "bike-uci-$([Guid]::NewGuid().ToString('N'))"
try {
    Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip" `
        -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $expand -Force
    $found = Get-ChildItem -Path $expand -Recurse -Filter "hour.csv" | Select-Object -First 1
    if (-not $found) { throw "hour.csv not found inside UCI zip." }
    Copy-Item $found.FullName $dest -Force
    $h = (Get-FileHash $dest -Algorithm MD5).Hash
    if ($h -ne "D50BD5A6F55131E72A7BEDC334E2FCE1") {
        throw "MD5 mismatch (got $h). Remove $dest and investigate."
    }
    Write-Host "OK: $dest (md5 matches hour.csv.dvc)"
}
finally {
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Remove-Item $expand -Recurse -Force -ErrorAction SilentlyContinue
}
