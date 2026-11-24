# Build one article table per extra event from the downloaded year files.
# Waits for faro_1918 to finish downloading (download_years.ps1 logs "done 1918").
$root = Join-Path $PSScriptRoot '..'
Set-Location $root
$py = '.venv\Scripts\python.exe'
$log = 'data\build_events.log'
$events = @(
  @{year=1929; start='1929-10-18'; end='1929-11-02'; event='crash-1929'},
  @{year=1938; start='1938-09-24'; end='1938-10-04'; event='munich-1938'},
  @{year=1941; start='1941-12-03'; end='1941-12-12'; event='pearl-harbor-1941'},
  @{year=1918; start='1918-11-04'; end='1918-11-14'; event='armistice-1918'}
)
foreach ($e in $events) {
  if ($e.year -eq 1918) {
    while (-not (Select-String -Path 'data\download.log' -Pattern 'done 1918' -Quiet)) { Start-Sleep -Seconds 60 }
  }
  $out = "data\articles_$($e.event).parquet"
  Add-Content $log "$(Get-Date -Format HH:mm:ss) start $($e.event)"
  & $py scripts\build_articles.py --year $e.year --start $e.start --end $e.end --event $e.event --out $out *>> $log
  Add-Content $log "$(Get-Date -Format HH:mm:ss) done $($e.event) exit=$LASTEXITCODE"
}
