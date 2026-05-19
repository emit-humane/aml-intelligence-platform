# ============================================================
# send_demo_transactions.ps1
# Posts REAL labeled transactions to the live API and shows the
# fraud-vs-normal score separation.
#
# Why real rows (not fabricated ones): the scoring pipeline is
# stateful + graph-based. The GNN (78% weight), graph_boost and the
# behavioral rolling windows only have signal for accounts that exist
# in the seeded history / offline graph and for timestamps in the
# seeded era (2026-01 .. 2026-04). The previous version of this script
# invented accounts (ACC_STRUCT_01, OFFSHR_01, ...) with 2024
# timestamps, so the strong detectors were blind on the fraud rows and
# fraud/normal scored the same. Sampling real rows from the actual
# datasets fixes that.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\send_demo_transactions.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\send_demo_transactions.ps1 -BaseUrl http://localhost:8000 -Normal 60 -Fraud 20
# ============================================================

param(
    [string] $BaseUrl = "https://aml-intelligence-platform.onrender.com",
    [int]    $Normal  = 60,
    [int]    $Fraud   = 20,
    [int]    $AlertThreshold = 57          # v5 calibrated alert threshold
)

$URL        = "$BaseUrl/stream/transaction"
$HDR        = @{ "Content-Type" = "application/json" }
$STREAM_CSV = Join-Path $PSScriptRoot "..\data\raw\stream_transactions.csv"
$TRUTH_CSV  = Join-Path $PSScriptRoot "..\data\raw\hidden_ground_truth.csv"

foreach ($p in @($STREAM_CSV, $TRUTH_CSV)) {
    if (-not (Test-Path $p)) { Write-Host "Missing data file: $p" -ForegroundColor Red; exit 1 }
}

# Fields the API expects (StreamIngestion coerces types server-side)
$COLS = @(
    "transaction_id","timestamp","sender_account","receiver_account",
    "sender_name","receiver_name","sender_bank","receiver_bank",
    "sender_country","receiver_country","amount","currency",
    "transaction_type","payment_channel","device_id","ip_address",
    "geo_latitude","geo_longitude","merchant_category","transaction_status",
    "sender_balance_before","sender_balance_after",
    "receiver_balance_before","receiver_balance_after",
    "kyc_level","is_international","remarks"
)

function To-Payload($row) {
    $h = @{}
    foreach ($c in $COLS) {
        $v = $row.$c
        switch -regex ($c) {
            '^(amount|geo_latitude|geo_longitude|.*balance.*)$' { $h[$c] = [double]$v; break }
            '^kyc_level$'        { $h[$c] = [int]$v; break }
            '^is_international$'  { $h[$c] = ($v -eq "True" -or $v -eq "true" -or $v -eq "1"); break }
            default              { $h[$c] = "$v" }
        }
    }
    return $h
}

function Send-Tx($body) {
    try {
        return Invoke-RestMethod -Uri $URL -Method Post -Body ($body | ConvertTo-Json) -Headers $HDR -TimeoutSec 60
    } catch { return $null }
}

# ── Build the labeled sample ─────────────────────────────────────────────────
$truthIds = @{}
Import-Csv $TRUTH_CSV | ForEach-Object { $truthIds[$_.transaction_id] = $true }

$fraudRows  = Import-Csv $TRUTH_CSV | Get-Random -Count $Fraud
$normalRows = Import-Csv $STREAM_CSV |
    Where-Object { -not $truthIds.ContainsKey($_.transaction_id) } |
    Get-Random -Count $Normal

$normScores  = New-Object System.Collections.ArrayList
$fraudScores = New-Object System.Collections.ArrayList
$ok = 0; $fail = 0

Write-Host "`n=== $Normal NORMAL transactions (real, seeded accounts) ===" -ForegroundColor Cyan
foreach ($row in $normalRows) {
    $resp = Send-Tx (To-Payload $row)
    if ($resp) {
        $ok++; [void]$normScores.Add([double]$resp.transaction_risk_score)
        $col = if ($resp.transaction_risk_score -ge $AlertThreshold) { "Yellow" } else { "Green" }
        Write-Host ("  {0}  score={1,6:N1}  {2}" -f `
            $row.transaction_id, $resp.transaction_risk_score, $resp.risk_level) -ForegroundColor $col
    } else { $fail++; Write-Host "  $($row.transaction_id)  FAILED" -ForegroundColor Red }
}

Write-Host "`n=== $Fraud FRAUD transactions (ground truth) ===" -ForegroundColor Cyan
foreach ($row in $fraudRows) {
    $resp = Send-Tx (To-Payload $row)
    if ($resp) {
        $ok++; [void]$fraudScores.Add([double]$resp.transaction_risk_score)
        $col = if ($resp.transaction_risk_score -ge $AlertThreshold) { "Red" } else { "DarkYellow" }
        Write-Host ("  {0}  score={1,6:N1}  {2}  [{3}]" -f `
            $row.transaction_id, $resp.transaction_risk_score, $resp.risk_level, $row.synthetic_pattern_type) `
            -ForegroundColor $col
    } else { $fail++; Write-Host "  $($row.transaction_id)  FAILED" -ForegroundColor Red }
}

# ── Separation summary ───────────────────────────────────────────────────────
function Stat($arr) {
    if ($arr.Count -eq 0) { return @{ n=0; mean=0; med=0 } }
    $s = $arr | Sort-Object
    $mean = ($arr | Measure-Object -Average).Average
    $med  = $s[[int][math]::Floor($s.Count/2)]
    return @{ n=$arr.Count; mean=$mean; med=$med }
}
$ns = Stat $normScores; $fs = Stat $fraudScores
$nFlag = ($normScores  | Where-Object { $_ -ge $AlertThreshold }).Count
$fFlag = ($fraudScores | Where-Object { $_ -ge $AlertThreshold }).Count

Write-Host "`n$("=" * 60)" -ForegroundColor Cyan
Write-Host ("sent={0} failed={1}   alert threshold={2}" -f $ok, $fail, $AlertThreshold) -ForegroundColor White
Write-Host ("NORMAL  n={0,3}  mean={1,6:N1}  median={2,6:N1}  alerted={3}/{0}" -f `
    $ns.n, $ns.mean, $ns.med, $nFlag) -ForegroundColor Green
Write-Host ("FRAUD   n={0,3}  mean={1,6:N1}  median={2,6:N1}  alerted={3}/{0}" -f `
    $fs.n, $fs.mean, $fs.med, $fFlag) -ForegroundColor Red
Write-Host ("SEPARATION  fraud median - normal median = {0:N1} / 100" -f `
    ($fs.med - $ns.med)) -ForegroundColor White
Write-Host ("(calibrated score = P(fraud)*100; expect normal ~0-20, fraud ~85-99)") -ForegroundColor DarkGray
