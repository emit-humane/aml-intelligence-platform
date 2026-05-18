# demo_layer2_behavioral.ps1 -- Layer 2 Behavioral Anomaly Detection Demo
# Tests 7 distinct behavioral scenarios, each targeting different feature dims
#
# Layer 2 ensemble = (IsoForest + LOF + Autoencoder) / 3
# Fires "behavioral_anomaly" when ensemble_anomaly_score > 60
# anomaly_drivers = top-3 feature names by absolute SCALED value
#
# Fusion: final_score = 0.30*rule + 0.25*behavioral + 0.20*gnn + 0.25*graph_boost
#
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\demo_layer2_behavioral.ps1

$URL = "https://aml-intelligence-platform.onrender.com/stream/transaction"
$HDR = @{ "Content-Type" = "application/json" }
$RUN = [System.Guid]::NewGuid().ToString("N").Substring(0, 4).ToUpper()
Write-Host ""
Write-Host "  Run ID: $RUN" -ForegroundColor DarkGray

function Tx {
    param(
        [string]$id,
        [string]$sender,
        [string]$receiver,
        [double]$amount,
        [string]$ts,
        [string]$type = "NEFT",
        [string]$channel = "Web",
        [string]$srcCountry = "IN",
        [string]$dstCountry = "IN",
        [bool]$intl = $false,
        [int]$kyc = 2,
        [double]$lat = 20.0,
        [double]$lon = 78.0,
        [string]$device = "STD_DEV",
        [double]$balBefore = -1,
        [string]$remarks = ""
    )
    if ($balBefore -lt 0) { $balBefore = [double](Get-Random -Min 2000000 -Max 8000000) }
    $balAfter = [double]($balBefore - $amount)
    return @{
        transaction_id          = "${id}_${RUN}"
        sender_account          = "${sender}_${RUN}"
        receiver_account        = "${receiver}_${RUN}"
        sender_name             = "Demo_${sender}"
        receiver_name           = "Demo_${receiver}"
        amount                  = $amount
        currency                = "INR"
        timestamp               = $ts
        transaction_type        = $type
        payment_channel         = $channel
        sender_country          = $srcCountry
        receiver_country        = $dstCountry
        sender_bank             = "HDFC"
        receiver_bank           = "SBI"
        transaction_status      = "Success"
        is_international        = $intl
        kyc_level               = $kyc
        geo_latitude            = $lat
        geo_longitude           = $lon
        sender_balance_before   = $balBefore
        sender_balance_after    = $balAfter
        receiver_balance_before = 50000.0
        receiver_balance_after  = [double](50000.0 + $amount)
        device_id               = "${device}_${RUN}"
        ip_address              = ""
        remarks                 = $remarks
        merchant_category       = ""
    }
}

function Post-Behav($body, [string]$label = "", [bool]$verbose = $false) {
    try {
        $resp = Invoke-RestMethod -Uri $URL -Method Post `
                -Body ($body | ConvertTo-Json) -Headers $HDR

        $finalScore  = $resp.transaction_risk_score
        $level       = $resp.risk_level
        $ruleScore   = $resp.score_breakdown.rule
        $behavScore  = $resp.score_breakdown.behavioral
        $gnnScore    = $resp.score_breakdown.gnn
        $isoScore    = [math]::Round($resp.behav_out.iso_forest_score, 1)
        $lofScore    = [math]::Round($resp.behav_out.lof_score, 1)
        $aeScore     = [math]::Round($resp.behav_out.autoencoder_score, 1)
        $ensemble    = [math]::Round($resp.behav_out.ensemble_anomaly_score, 1)
        $drivers     = ($resp.behav_out.anomaly_drivers) -join " | "
        $patterns    = ($resp.triggered_patterns) -join ", "
        $explanation = $resp.explanation

        $col = if ($level -eq "Critical") { "Magenta" } `
          elseif ($level -eq "High")      { "Red"     } `
          elseif ($level -eq "Medium")    { "Yellow"  } `
          else                            { "Green"   }

        Write-Host ("  {0,-30} final={1,5}  behav={2,5}  ensemble={3,5}  level={4,-8}" `
            -f $label, $finalScore, $behavScore, $ensemble, $level) -ForegroundColor $col
        Write-Host ("  {0,-30} iso={1,5}  lof={2,5}  ae={3,5}  drivers=[{4}]" `
            -f "", $isoScore, $lofScore, $aeScore, $drivers) -ForegroundColor DarkCyan
        if ($verbose) {
            Write-Host ("  {0,-30} patterns=[{1}]" -f "", $patterns) -ForegroundColor DarkGray
            Write-Host ("  {0,-30} {1}" -f "", $explanation) -ForegroundColor DarkGray
        }
        Write-Host ""
        return $resp
    } catch {
        $s = $_.Exception.Response
        if ($s) {
            $r   = [System.IO.StreamReader]::new($s.GetResponseStream())
            $msg = "HTTP $($s.StatusCode.value__): $($r.ReadToEnd())"
        } else {
            $msg = $_.Exception.Message
        }
        Write-Host "  $label  ERROR: $msg" -ForegroundColor DarkRed
        Write-Host ""
        return $null
    }
}

Write-Host ""
Write-Host ("=" * 75) -ForegroundColor Cyan
Write-Host "  Layer 2 -- Behavioral Anomaly Detection Demo" -ForegroundColor Cyan
Write-Host "  Ensemble = (IsoForest + LOF + Autoencoder) / 3   fires when > 60" -ForegroundColor Cyan
Write-Host "  anomaly_drivers = top-3 features by absolute SCALED value" -ForegroundColor Cyan
Write-Host ("=" * 75) -ForegroundColor Cyan
Write-Host ""


# ================================================================
# B00 -- CONTROL (normal baseline)
# Shows what a fresh-account normal transaction looks like.
# No rules fired, behavioral is the only signal.
# ================================================================
Write-Host "[B00] CONTROL -- Normal baseline transaction" -ForegroundColor White

Post-Behav (Tx -id "B00_001" -sender "B00_NORMAL" -receiver "B00_RX1" `
               -amount 50000 -ts "2024-07-10T10:00:00Z") "B00_normal_50K" $true


# ================================================================
# B01 -- AMOUNT Z-SCORE SPIKE
# Feature targeted: amount_zscore  (index 14 in behavioral cols)
# amount_zscore = (amount - avg_7d) / (std_7d + 1e-6)
#
# Build 20 small transactions (~20K each) to set avg_7d and std_7d,
# then send ONE transaction 100x larger -> amount_zscore >> 50
# Behavioral driver should flip to "amount_zscore"
# ================================================================
Write-Host "[B01] AMOUNT Z-SCORE SPIKE  (build 20x 20K history, then spike to 2,000,000)" -ForegroundColor Magenta

# Seed: 20 transactions of ~20K spread over 4 days to build avg_7d
$b01amounts = @(18000,22000,19500,21000,20500,17800,23000,20000,19000,21500,
                18500,22500,19800,20200,21200,18200,22800,19200,20800,21800)
for ($j = 0; $j -lt 20; $j++) {
    $day  = "{0:D2}" -f [int]([math]::Floor($j / 5) + 1)
    $hr   = "{0:D2}" -f [int](($j % 5) * 3 + 8)
    $txid = "B01_SEED_{0:D3}" -f ($j + 1)
    Post-Behav (Tx -id $txid -sender "B01_ACC" -receiver "B01_RX" `
                   -amount $b01amounts[$j] -ts "2024-07-${day}T${hr}:00:00Z") "" $false | Out-Null
}
Write-Host "  [20 seed transactions sent at ~20K each -- avg_7d established]" -ForegroundColor DarkGray
Write-Host ""

# Trigger: 2,000,000 -- amount_zscore = (2M - 20K) / std >> 50 sigma
Post-Behav (Tx -id "B01_SPIKE" -sender "B01_ACC" -receiver "B01_RX_BIG" `
               -amount 2000000 -ts "2024-07-05T20:00:00Z") "B01_TRIGGER(2M_vs_20K_avg)" $true


# ================================================================
# B02 -- VELOCITY BURST + TINY GAPS
# Features targeted: tx_velocity_1h (index 0), tx_gap_seconds (index 6)
#
# 15 transactions from same account in 7 minutes (every 30 seconds)
# tx_velocity_1h = 14 on the 15th tx (pre-event count)
# tx_gap_seconds = 30s (vs. normal hours/days between txns)
# ================================================================
Write-Host "[B02] VELOCITY BURST  (15 txns in 7 min, 30s apart -- v1h=14, gap=30s)" -ForegroundColor Magenta

for ($j = 0; $j -lt 15; $j++) {
    $totSec = $j * 30
    $mm  = "{0:D2}" -f [int]([math]::Floor($totSec / 60))
    $ss  = "{0:D2}" -f [int]($totSec % 60)
    $txid = "B02_{0:D3}" -f ($j + 1)
    $recv = "B02_RX{0:D2}" -f ($j + 1)
    if ($j -eq 14) {
        Post-Behav (Tx -id $txid -sender "B02_ACC" -receiver $recv `
                       -amount 30000 -ts "2024-07-10T09:${mm}:${ss}Z") "B02_TRIGGER(pre-v=14)" $true
    } else {
        Post-Behav (Tx -id $txid -sender "B02_ACC" -receiver $recv `
                       -amount 30000 -ts "2024-07-10T09:${mm}:${ss}Z") "B02_seed$($j+1)(v=$j)" $false | Out-Null
    }
}


# ================================================================
# B03 -- BALANCE DRAIN
# Feature targeted: balance_utilization (index 31)
# balance_utilization = amount / sender_balance_before
#
# Account has 10,000,000 balance; sends 9,700,000 in one shot.
# balance_utilization = 0.97  (training avg likely 0.05-0.20)
# Also triggers R01 (amount > 1M) -- shows multi-layer detection
# ================================================================
Write-Host "[B03] BALANCE DRAIN  (9.7M out of 10M balance -- util=0.97)" -ForegroundColor Magenta

Post-Behav (Tx -id "B03_001" -sender "B03_ACC" -receiver "B03_RX1" `
               -amount 9700000 -ts "2024-07-10T10:30:00Z" `
               -balBefore 10000000) "B03_TRIGGER(util=0.97)" $true


# ================================================================
# B04 -- NIGHT ACTIVITY SHIFT
# Feature targeted: night_tx_ratio (index 7), hour_of_day (index 9)
# night_tx_ratio = fraction of 7d txns at hours 22:00-05:59
#
# Step 1: Build 10 normal daytime transactions (09:00-17:00)
# Step 2: 5 deep-night transactions at 02:00-04:00
# night_tx_ratio shifts from 0 -> 5/15 = 0.33 on trigger
# ================================================================
Write-Host "[B04] NIGHT ACTIVITY SHIFT  (daytime history, then 02:00-04:00 txns)" -ForegroundColor Magenta

# Seed: 10 daytime transactions
$dayHours = @("09","10","11","12","13","14","15","16","09","11")
for ($j = 0; $j -lt 10; $j++) {
    $txid = "B04_DAY_{0:D3}" -f ($j + 1)
    Post-Behav (Tx -id $txid -sender "B04_ACC" -receiver "B04_RX" `
                   -amount 40000 -ts "2024-07-08T$($dayHours[$j]):00:00Z") "" $false | Out-Null
}
Write-Host "  [10 daytime seed transactions sent (09:00-16:00)]" -ForegroundColor DarkGray
Write-Host ""

# Night triggers
$nightHours = @("02","02","03","03","04")
for ($j = 0; $j -lt 5; $j++) {
    $min  = "{0:D2}" -f ($j * 12)
    $txid = "B04_NIGHT_{0:D3}" -f ($j + 1)
    Post-Behav (Tx -id $txid -sender "B04_ACC" -receiver "B04_RX_NIGHT" `
                   -amount 40000 -ts "2024-07-09T$($nightHours[$j]):${min}:00Z") "B04_night_tx$($j+1)" $true
}


# ================================================================
# B05 -- DEVICE HOPPING (account takeover signal)
# Features targeted: device_change_count_7d (index 25), new_device_flag (index 26)
# device_change_count_7d = unique device IDs used in last 7d (pre-event)
#
# Same account uses 8 completely different devices.
# new_device_flag = 1 on every single transaction.
# device_change_count_7d rises 0 -> 1 -> 2 -> ... -> 7 on trigger.
# ================================================================
Write-Host "[B05] DEVICE HOPPING  (8 txns, each from a different new device)" -ForegroundColor Magenta

1..8 | ForEach-Object {
    $i   = $_
    $dev = "B05_DEV${i}"
    $txid = "B05_{0:D3}" -f $i
    if ($i -eq 8) {
        Post-Behav (Tx -id $txid -sender "B05_ACC" -receiver "B05_RX" `
                       -amount 150000 -ts "2024-07-10T$("{0:D2}" -f ($i+8)):00:00Z" `
                       -device $dev) "B05_TRIGGER(dev_cnt=7)" $true
    } else {
        Post-Behav (Tx -id $txid -sender "B05_ACC" -receiver "B05_RX" `
                       -amount 150000 -ts "2024-07-10T$("{0:D2}" -f ($i+8)):00:00Z" `
                       -device $dev) "B05_seed${i}(dev_cnt=$($i-1))" $false | Out-Null
    }
}


# ================================================================
# B06 -- INTERNATIONAL COUNTRY SCATTER (geographic risk)
# Features targeted: high_risk_country_flag (index 23),
#                    country_switch_count_7d (index 21),
#                    cross_border_ratio_30d (index 24)
#
# Account sends to 6 different high-risk countries in quick succession.
# Each tx: is_international=True, dstCountry in {AE,NG,PK,CN,MU,CH}
# country_switch_count_7d = 5 on the 6th tx
# ================================================================
Write-Host "[B06] COUNTRY SCATTER  (6 high-risk international txns to AE/NG/PK/CN/MU/CH)" -ForegroundColor Magenta

$countries = @("AE","NG","PK","CN","MU","CH")
$countryNames = @("UAE","Nigeria","Pakistan","China","Mauritius","Switzerland")
for ($j = 0; $j -lt 6; $j++) {
    $txid = "B06_{0:D3}" -f ($j + 1)
    $hr   = "{0:D2}" -f ($j * 2 + 8)
    $label = "B06_tx$($j+1)_$($countryNames[$j])"
    Post-Behav (Tx -id $txid -sender "B06_ACC" -receiver "B06_OFFSHR" `
                   -amount 500000 -ts "2024-07-10T${hr}:00:00Z" `
                   -type "Wire" -srcCountry "IN" -dstCountry $countries[$j] `
                   -intl $true -kyc 1) $label $true
}


# ================================================================
# B07 -- COMBINED MULTI-SIGNAL (max behavioral pressure)
# All high-risk features combined in a single transaction:
#   - Large amount  -> amount_zscore high if history exists
#   - New device    -> new_device_flag = 1
#   - Night time    -> night activity (01:30)
#   - High-risk country + international
#   - Balance drain -> balance_utilization = 0.98
#   - KYC=0         -> triggers R12 rule as well
# This should push ALL three models (iso, lof, ae) to maximum
# ================================================================
Write-Host "[B07] COMBINED MULTI-SIGNAL  (new device + night + intl HR country + bal drain + kyc=0)" -ForegroundColor Magenta

$devB07 = "B07_HIJACKED_DEV"
Post-Behav (Tx -id "B07_001" -sender "B07_ACC" -receiver "B07_OFFSHR" `
               -amount 4800000 -ts "2024-07-10T01:30:00Z" `
               -type "Wire" -srcCountry "IN" -dstCountry "AE" `
               -intl $true -kyc 0 `
               -lat 19.07 -lon 72.88 `
               -device $devB07 `
               -balBefore 4900000) "B07_TRIGGER(all_signals)" $true


# ================================================================
Write-Host ("=" * 75) -ForegroundColor Cyan
Write-Host "  Summary -- what to look for:" -ForegroundColor Yellow
Write-Host "  B00: baseline -- behavioral always fires (LOF+AE saturated on fresh accts)" -ForegroundColor Yellow
Write-Host "  B01: amount_zscore should appear in drivers after history built" -ForegroundColor Yellow
Write-Host "  B02: tx_velocity_1h / tx_gap_seconds should dominate drivers" -ForegroundColor Yellow
Write-Host "  B03: balance_utilization should appear in drivers" -ForegroundColor Yellow
Write-Host "  B04: night_tx_ratio / hour_of_day should appear in drivers" -ForegroundColor Yellow
Write-Host "  B05: device_change_count_7d / new_device_flag should appear" -ForegroundColor Yellow
Write-Host "  B06: high_risk_country_flag / country_switch_count_7d should appear" -ForegroundColor Yellow
Write-Host "  B07: iso_forest_score highest of all -- all 52 features anomalous" -ForegroundColor Yellow
Write-Host ("=" * 75) -ForegroundColor Cyan
