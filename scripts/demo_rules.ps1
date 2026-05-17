# =============================================================
# demo_rules.ps1  —  One fraud scenario per rule, R01-R15
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\demo_rules.ps1
#
# Each run generates a unique 4-char suffix so account/device IDs are
# fresh every time, avoiding Render's persistent feature-store state
# from polluting the expected trigger sequence.
# =============================================================

$URL  = "https://aml-intelligence-platform.onrender.com/stream/transaction"
$HDR  = @{ "Content-Type" = "application/json" }

# Unique suffix per run — prevents feature-store pollution across runs
$RUN = [System.Guid]::NewGuid().ToString("N").Substring(0, 4).ToUpper()
Write-Host "`n  Run ID: $RUN  (all account/device IDs are suffixed with this)" -ForegroundColor DarkGray

function Tx {
    param(
        [string]$id, [string]$sender, [string]$receiver,
        [double]$amount, [string]$ts,
        [string]$type="NEFT", [string]$channel="Web",
        [string]$srcCountry="IN", [string]$dstCountry="IN",
        [bool]$intl=$false, [int]$kyc=2,
        [double]$lat=20.0, [double]$lon=78.0,
        [string]$device="", [string]$remarks=""
    )
    $bal = [double](Get-Random -Min 500000 -Max 5000000)
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
        sender_balance_before   = $bal
        sender_balance_after    = [double]($bal - $amount)
        receiver_balance_before = [double](Get-Random -Min 10000 -Max 200000)
        receiver_balance_after  = [double](Get-Random -Min 10000 -Max 200000)
        device_id               = $device
        ip_address              = ""
        remarks                 = $remarks
        merchant_category       = ""
    }
}

function Post-Tx($body, [string]$label="") {
    try {
        $resp = Invoke-RestMethod -Uri $URL -Method Post `
                -Body ($body | ConvertTo-Json) -Headers $HDR
        $score = $resp.transaction_risk_score
        $rules = $resp.rule_out.triggered_rules -join ", "
        $level = $resp.risk_level
        $col   = switch ($level) {
            "Critical" { "Magenta" }; "High" { "Red" }
            "Medium"   { "Yellow"  }; default { "Green" }
        }
        Write-Host ("  {0,-22} score={1,6}  level={2,-8}  rules=[{3}]" `
            -f $label, $score, $level, $rules) -ForegroundColor $col
        return $resp
    } catch {
        $s = $_.Exception.Response
        $msg = if ($s) {
            $r = [System.IO.StreamReader]::new($s.GetResponseStream())
            "HTTP $($s.StatusCode.value__): $($r.ReadToEnd())"
        } else { $_.Exception.Message }
        Write-Host "  $label  ERROR: $msg" -ForegroundColor DarkRed
        return $null
    }
}

Write-Host "`n$('='*68)" -ForegroundColor Cyan
Write-Host "  AML Rule Demo  —  R01 through R15" -ForegroundColor Cyan
Write-Host "$('='*68)`n" -ForegroundColor Cyan


# ─────────────────────────────────────────────────────────────
# R01  high_value_transfer   condition: amount > 1,000,000
# ─────────────────────────────────────────────────────────────
Write-Host "[R01] High-Value Transfer  (amount > 1,000,000)" -ForegroundColor Magenta

Post-Tx (Tx -id "R01_001" -sender "R01_SND" -receiver "R01_RCV" `
            -amount 1500000 -ts "2024-07-01T10:00:00Z") "R01_trigger(1.5M)"

Post-Tx (Tx -id "R01_002" -sender "R01_SND_NO" -receiver "R01_RCV" `
            -amount 999999 -ts "2024-07-01T10:01:00Z") "R01_no_trigger(999K)"


# ─────────────────────────────────────────────────────────────
# R02  structuring
#      condition: sub_threshold_flag AND tx_velocity_1h >= 3
#      sub_threshold = amount in [850_000, 1_000_000)
#      Send 3 sub-threshold txns from same account within 1 hour
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R02] Structuring  (3 sub-threshold txns in 1 hour)" -ForegroundColor Magenta

Post-Tx (Tx -id "R02_001" -sender "R02_SND" -receiver "R02_RX1" `
            -amount 899000 -ts "2024-07-02T09:00:00Z") "R02_setup_1(vel=1)"
Post-Tx (Tx -id "R02_002" -sender "R02_SND" -receiver "R02_RX2" `
            -amount 875000 -ts "2024-07-02T09:15:00Z") "R02_setup_2(vel=2)"
Post-Tx (Tx -id "R02_003" -sender "R02_SND" -receiver "R02_RX3" `
            -amount 950000 -ts "2024-07-02T09:30:00Z") "R02_trigger(vel=3)"


# ─────────────────────────────────────────────────────────────
# R03  velocity_spike
#      condition: tx_velocity_1h > 10 OR tx_velocity_24h > 50
#      Send 11 txns within 1 hour (fires on the 11th)
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R03] Velocity Spike  (11 txns in 1 hour)" -ForegroundColor Magenta

for ($j = 0; $j -lt 11; $j++) {
    $min   = "{0:D2}" -f ($j * 5)
    $txid  = "R03_{0:D3}" -f ($j + 1)
    $rxacc = "R03_RX{0:D2}" -f ($j + 1)
    $label = if ($j -eq 10) { "R03_trigger(vel=11)" } else { "R03_setup_$($j+1)(vel=$($j+1))" }
    Post-Tx (Tx -id $txid -sender "R03_SND" -receiver $rxacc `
                -amount 30000 -ts "2024-07-03T08:${min}:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R04  dormant_activation
#      condition: tx_gap_seconds > 15,552,000 AND amount > 100,000
#      15,552,000 s = 180 days
#      Step 1: seed with an old timestamp (> 180 days ago)
#      Step 2: send a new transaction — gap computed from last_tx_ts_ever
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R04] Dormant Activation  (gap > 180 days, amount > 100K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R04_001" -sender "R04_SND" -receiver "R04_RX1" `
            -amount 5000 -ts "2023-11-01T10:00:00Z") "R04_dormant_seed(2023)"
Post-Tx (Tx -id "R04_002" -sender "R04_SND" -receiver "R04_RX2" `
            -amount 200000 -ts "2024-07-01T10:00:00Z") "R04_trigger(242d gap)"


# ─────────────────────────────────────────────────────────────
# R05  impossible_travel
#      condition: impossible_travel_flag == True  (speed > 900 km/h)
#      Mumbai (19.07, 72.88) -> London (51.51, -0.13) in 30 min
#      Distance ~7,200 km / 0.5 h = 14,400 km/h >> 900
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R05] Impossible Travel  (Mumbai -> London in 30 min)" -ForegroundColor Magenta

Post-Tx (Tx -id "R05_001" -sender "R05_SND" -receiver "R05_RX1" `
            -amount 10000 -ts "2024-07-04T10:00:00Z" `
            -lat 19.07 -lon 72.88) "R05_Mumbai(seed)"
Post-Tx (Tx -id "R05_002" -sender "R05_SND" -receiver "R05_RX2" `
            -amount 10000 -ts "2024-07-04T10:30:00Z" `
            -lat 51.51 -lon (-0.13)) "R05_trigger(London,30min)"


# ─────────────────────────────────────────────────────────────
# R06  high_risk_jurisdiction
#      condition: high_risk_country_flag AND is_international == True
#      High-risk countries: AE, MU, CN, NG, PK, CH
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R06] High-Risk Jurisdiction  (IN -> PK, international)" -ForegroundColor Magenta

Post-Tx (Tx -id "R06_001" -sender "R06_SND" -receiver "R06_OFFSHR" `
            -amount 800000 -ts "2024-07-04T11:00:00Z" `
            -type "Wire" -srcCountry "IN" -dstCountry "PK" `
            -intl $true -kyc 1) "R06_trigger(PK,intl)"

Post-Tx (Tx -id "R06_002" -sender "R06_SND_NO" -receiver "R06_DOM" `
            -amount 800000 -ts "2024-07-04T11:01:00Z" `
            -srcCountry "IN" -dstCountry "IN" -intl $false) "R06_no_trigger(domestic)"


# ─────────────────────────────────────────────────────────────
# R07  device_anomaly
#      condition: new_device_flag AND amount > 200,000
#      First-ever tx from this account uses a brand-new device
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R07] Device Anomaly  (new device + amount > 200K)" -ForegroundColor Magenta

$devR07 = "DEV_R07_NEW_${RUN}"

Post-Tx (Tx -id "R07_001" -sender "R07_SND" -receiver "R07_RX1" `
            -amount 350000 -ts "2024-07-04T12:00:00Z" `
            -device $devR07) "R07_trigger(new_dev)"

Post-Tx (Tx -id "R07_002" -sender "R07_SND" -receiver "R07_RX2" `
            -amount 350000 -ts "2024-07-04T12:05:00Z" `
            -device $devR07) "R07_no_trigger(same_dev)"


# ─────────────────────────────────────────────────────────────
# R08  shared_device
#      condition: shared_device_count > 5
#      One device used by 7 distinct accounts → fires on 7th
#      (shared_device_count is the PRE-event count, so 6 must be
#       ingested before the 7th tx is evaluated)
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R08] Shared Device  (1 device used by 7 accounts)" -ForegroundColor Magenta

$devR08 = "DEV_SHARED_R08_${RUN}"
1..7 | ForEach-Object {
    $i     = $_
    $txid  = "R08_{0:D3}" -f $i
    $acct  = "R08_MULE_{0:D2}" -f $i
    $label = if ($i -le 6) { "R08_setup(acct$i,cnt=$($i-1))" } else { "R08_trigger(acct7,cnt=6)" }
    Post-Tx (Tx -id $txid -sender $acct -receiver "R08_HUB" `
                -amount 50000 -ts "2024-07-04T13:0${i}:00Z" `
                -device $devR08) $label
}


# ─────────────────────────────────────────────────────────────
# R09  excessive_beneficiaries
#      condition: beneficiary_count_7d > 20
#      One sender pays 21 distinct receivers within 7 days
#      Fires on the 21st unique receiver
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R09] Excessive Beneficiaries  (21 unique receivers in 7 days)" -ForegroundColor Magenta

1..21 | ForEach-Object {
    $i    = $_
    $txid = "R09_{0:D3}" -f $i
    $recv = "R09_RECV_{0:D2}" -f $i
    # IMPORTANT: cast Floor result to [int] — {D:2} format spec requires integer
    $day  = "{0:D2}" -f [int]([math]::Floor(($i - 1) / 4) + 1)
    $hr   = "{0:D2}" -f [int]((($i - 1) % 4) * 6)
    $label = if ($i -eq 21) { "R09_trigger(recv21,cnt=21)" } else { "R09_setup(recv$i)" }
    Post-Tx (Tx -id $txid -sender "R09_SND" -receiver $recv `
                -amount 5000 -ts "2024-07-${day}T${hr}:00:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R10  cycle_closure
#      condition: edge_creates_cycle == True
#      Build A -> B -> C -> A — 3rd tx closes the cycle
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R10] Cycle Closure  (A->B->C->A circular laundering)" -ForegroundColor Magenta

Post-Tx (Tx -id "R10_001" -sender "R10_A" -receiver "R10_B" `
            -amount 500000 -ts "2024-07-04T14:00:00Z") "R10_A->B(no_cycle)"
Post-Tx (Tx -id "R10_002" -sender "R10_B" -receiver "R10_C" `
            -amount 490000 -ts "2024-07-04T14:30:00Z") "R10_B->C(no_cycle)"
Post-Tx (Tx -id "R10_003" -sender "R10_C" -receiver "R10_A" `
            -amount 480000 -ts "2024-07-04T15:00:00Z") "R10_trigger(C->A,cycle!)"


# ─────────────────────────────────────────────────────────────
# R11  round_amount_structuring
#      condition: round_amount_flag AND tx_velocity_24h > 5
#      round_amount_flag = amount is multiple of 100,000
#      Send 6 round-amount txns within 24h; fires on the 6th
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R11] Round Amount Structuring  (6 round-amount txns in 24h)" -ForegroundColor Magenta

$r11amounts = @(300000, 500000, 200000, 100000, 400000, 600000)
for ($j = 0; $j -lt 6; $j++) {
    $hr    = "{0:D2}" -f ($j * 3)
    $txid  = "R11_{0:D3}" -f ($j + 1)
    $rxacc = "R11_RX_{0:D2}" -f ($j + 1)
    $label = if ($j -eq 5) { "R11_trigger(vel=6)" } else { "R11_setup_$($j+1)(vel=$($j+1))" }
    Post-Tx (Tx -id $txid -sender "R11_SND" -receiver $rxacc `
                -amount $r11amounts[$j] `
                -ts "2024-07-04T${hr}:00:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R12  kyc_mismatch
#      condition: kyc_level == 0 AND amount > 500,000
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R12] KYC Mismatch  (kyc_level=0, amount > 500K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R12_001" -sender "R12_SND" -receiver "R12_RX1" `
            -amount 750000 -ts "2024-07-04T16:00:00Z" -kyc 0) "R12_trigger(kyc=0)"

Post-Tx (Tx -id "R12_002" -sender "R12_SND_NO" -receiver "R12_RX1" `
            -amount 750000 -ts "2024-07-04T16:01:00Z" -kyc 1) "R12_no_trigger(kyc=1)"


# ─────────────────────────────────────────────────────────────
# R13  benford_anomaly
#      condition: benford_chi2_score > 3.84  (min 5 txns required)
#      All amounts start with digit 9 — extreme Benford violation
#      Benford expects ~4.6% for digit 9; sending 100% gives chi2 >> 3.84
#      Fires from the 5th transaction onward
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R13] Benford Anomaly  (15 txns, all amounts start with digit 9)" -ForegroundColor Magenta

$r13amounts = @(90000,91000,92000,93000,94000,95000,96000,97000,98000,99000,
                90500,91500,92500,93500,94500)
for ($j = 0; $j -lt 15; $j++) {
    $hr    = "{0:D2}" -f ($j + 1)
    $txid  = "R13_{0:D3}" -f ($j + 1)
    $rxacc = "R13_RX_{0:D2}" -f ($j + 1)
    $label = if ($j -ge 4) { "R13_trigger(tx$($j+1),chi2>3.84)" } else { "R13_setup_$($j+1)" }
    Post-Tx (Tx -id $txid -sender "R13_SND" -receiver $rxacc `
                -amount $r13amounts[$j] -ts "2024-07-04T${hr}:00:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R14  fan_in_collector
#      condition: receiver_in_degree_unique >= 3
#      receiver_in_degree_unique is the PRE-event graph count, so
#      3 senders must be ingested BEFORE the 4th tx is evaluated
#      → send 4 transactions; R14 fires on the 4th
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R14] Fan-In Collector  (4 unique senders -> 1 collector)" -ForegroundColor Magenta

1..4 | ForEach-Object {
    $i     = $_
    $txid  = "R14_{0:D3}" -f $i
    $src   = "R14_SRC_{0:D2}" -f $i
    $label = if ($i -eq 4) { "R14_trigger(4th,indeg=3)" } else { "R14_setup(sender$i,indeg=$($i-1))" }
    Post-Tx (Tx -id $txid -sender $src -receiver "R14_COLLECT" `
                -amount 200000 -ts "2024-07-04T1${i}:00:00Z") $label
}


# ─────────────────────────────────────────────────────────────
# R15  passthrough_layering
#      condition: sender_in_degree >= 1 AND amount >= 500,000
#      Step 1: seed R15_MID as a receiver (gives it in_degree = 1)
#      Step 2: R15_MID forwards >= 500K onward → R15 fires
# ─────────────────────────────────────────────────────────────
Write-Host "`n[R15] Pass-Through Layering  (receive, then forward >= 500K)" -ForegroundColor Magenta

Post-Tx (Tx -id "R15_001" -sender "R15_SRC" -receiver "R15_MID" `
            -amount 800000 -ts "2024-07-04T17:00:00Z") "R15_seed(MID_receives)"
Post-Tx (Tx -id "R15_002" -sender "R15_MID" -receiver "R15_DST" `
            -amount 750000 -ts "2024-07-04T17:30:00Z") "R15_trigger(MID_forwards)"


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
Write-Host "`n$('='*68)" -ForegroundColor Cyan
Write-Host "  All 15 rule demos sent.  Run ID: $RUN" -ForegroundColor Cyan
Write-Host "  Expected single-rule fires:" -ForegroundColor Yellow
Write-Host "    R01:trigger  R02:3rd  R03:11th  R04:2nd  R05:London" -ForegroundColor Yellow
Write-Host "    R06:PK-wire  R07:1st  R08:7th   R09:21st R10:C->A" -ForegroundColor Yellow
Write-Host "    R11:6th      R12:1st  R13:5th+  R14:4th  R15:2nd" -ForegroundColor Yellow
Write-Host "$('='*68)" -ForegroundColor Cyan
